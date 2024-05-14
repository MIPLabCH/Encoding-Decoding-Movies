# -*- coding: utf-8 -*-

"""
This script contains all the functions related to the model, its training, and its validation
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show
from nilearn import plotting
from scipy.io import loadmat
from IPython.display import Image,FileLink, display, IFrame
import pandas as pd
import torch
import torch.nn as nn
import imageio
import time
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pytorch_msssim import ssim

from torchvision.transforms import Resize
from torchmetrics.image import TotalVariation
from torchvision.models import vgg16
from p3d_model import *
from dataset import *
from model import *

from dataset import * 
from visualisation import *


### ENCODER MODEL ###

class Encoder(nn.Module):
    def __init__(self, mask_size):
        """
        Encoder architecture. Inspired by the work of Kupershmidt et al. (2022).

        Parameters:
            mask_size (int): The number of voxels of the masked fMRI.
        """
        
        super(Encoder, self).__init__()

        # 3D convolutional layer 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0), padding_mode='zeros'),
            nn.MaxPool3d(kernel_size=(2, 2, 1))
        )

        # 3D convolutional layer 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 256, kernel_size=(1, 1, 5), padding=(0, 0, 2)),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.AvgPool3d((1, 1, 2))
        )

        # 16x1x1 temporal combinations
        num_combinations = 16
        self.temporal_combinations = nn.ModuleList()
        for _ in range(num_combinations):
            combination = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(128)
            )
            self.temporal_combinations.append(combination)

        # 2D convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # flatten + dropout
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5)    # hyperparameter ?
        )

        # fully connected layer
        self.fc = nn.Linear(12544, mask_size)

    def forward(self, x):
        """
        Forward pass of the Encoder.
        
        Parameters:
            x (tensor): Preprocessed video (input). Shape: (movie_duration, 3, 112, 112, 32).
        Returns: 
            x (tensor): Predicted fMRI signal (prediction). Shape: (movie_duration, mask_size)
        """
        
        # 3D convolutional layer 1
        x = self.conv1(x)

        # 3D convolutional layer 2
        x = self.conv2(x)

        # 16x1x1 temporal combinations
        tensor = []
        for i in range(len(self.temporal_combinations)):
            t = self.temporal_combinations[i](x[:, :, :, :, i])
            tensor.append(t)
        tensor = torch.cat(tensor, dim=1)

        # 2D convolutional layer
        x = self.conv3(tensor)

        # flatten + dropout
        x = self.flatten(x)

        # fully connected layer
        x = self.fc(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class E_Loss(nn.Module):
    def __init__(self, alpha=0.5, use_pretrained_decoder=False):
        """
        Encoder Loss Module with optional Decoder Loss.
        
        Parameters:
            alpha (float): Weight of the cosine similarity in the combined loss calculation.
            include_decoder_loss (bool): Flag to determine whether to include decoder loss in the loss computation.
        """
        super(E_Loss, self).__init__()
        self.alpha = alpha
        self.use_pretrained_decoder = use_pretrained_decoder
        if self.use_pretrained_decoder:
            self.d_loss = D_Loss()  # Assuming D_Loss is defined elsewhere

    def forward(self, encoder_prediction, encoder_label, decoder_prediction=None, decoder_label=None):
        """
        Calculate the Encoder loss, optionally including Decoder loss.
        
        Parameters:
            encoder_prediction (tensor): Output of the Encoder. Shape: (1, 4330).
            encoder_label (tensor): Ground truth (real fMRI data). Shape: (1, 4330).
            decoder_prediction (tensor, optional): Output of the Decoder, required if include_decoder_loss is True.
            decoder_label (tensor, optional): Ground truth for the Decoder output, required if include_decoder_loss is True.
        
        Returns:
            A tuple containing:
            - Cosine similarity mean (float)
            - Encoder loss (float)
            - Decoder loss (float, optional)
            - Total loss (float, includes decoder loss if applicable)
        """
        
        mse_loss = F.mse_loss(encoder_prediction, encoder_label) / encoder_label.shape[1]
        cos_sim = F.cosine_similarity(encoder_prediction, encoder_label, dim=1)
        e_loss = mse_loss + self.alpha * (1 - cos_sim).mean()

        if self.use_pretrained_decoder:
            if decoder_prediction is None or decoder_label is None:
                raise ValueError("Decoder predictions and labels must be provided if decoder loss is included.")
            _, _, _, d_loss, _ = self.d_loss(decoder_prediction, decoder_label)
            total_loss = e_loss + d_loss
            metrics_names = ['cos_sim', 'encoder_loss', 'decoder_loss', 'combined_loss']
            return cos_sim.mean().item(), e_loss.item(), d_loss.item(), total_loss, metrics_names
        else:
            metrics_names = ['cos_sim', 'encoder_loss']
            return cos_sim.mean().item(), e_loss, metrics_names

### DECODER MODEL ###

class Decoder(nn.Module):
    def __init__(self, mask_size):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        # Convolutional layer 1
        self.conv1 = nn.ConvTranspose2d(48, 48, kernel_size=5, stride=1, padding=2)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn1 = nn.BatchNorm2d(48)

        # Convolutional layer 2
        self.conv2 = nn.ConvTranspose2d(48, 48, kernel_size=5, stride=1, padding=2)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn2 = nn.BatchNorm2d(48)

        # Convolutional layer 3
        self.conv3 = nn.ConvTranspose2d(48, 48, kernel_size=5, stride=1, padding=2)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn3 = nn.BatchNorm2d(48)
        
        # Convolutional layer 4
        self.conv4 = nn.ConvTranspose2d(48, 3, kernel_size=5, stride=1, padding=2)
        #self.up4 = nn.Upsample(scale_factor=2, mode='nearest')  #other solution instead of removing the upsample is to put a padding of 30 but doesn't seem right
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, x):
        #print(x.shape)
        
        # Fully connected layer
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(-1, 48, 14, 14) # Reshape to (batch_size, channels, H, W)
        #print(x.shape)
        
        # First conv layer + ReLU + Upsample + BatchNorm
        x = self.conv1(x)
        x = F.relu(x)
        x = self.up1(x)
        x = self.bn1(x)
        #print(x.shape)

        # Second conv layer + ReLU + Upsample + BatchNorm
        x = self.conv2(x)
        x = F.relu(x)
        x = self.up2(x)
        x = self.bn2(x)
        #print(x.shape)

        # third conv layer + ReLU + Upsample + BatchNorm
        x = self.conv3(x)
        x = F.relu(x)
        x = self.up3(x)
        x = self.bn3(x)
        #print(x.shape)
        
        # fourth conv layer + sigmoid + Upsample + BatchNorm
        x = self.conv4(x)
        x = torch.sigmoid(x)
        #x = self.up4(x)
        x = self.bn4(x)
        #print(x.shape)
        
        return x


def normalize2(X):
    """
    Normalizes an array.
        Parameters:
            X (array): The numpy array to normalize. Any shape accepted.
        Returns:
            normalized_X (array): The normalized array.
    """
    
    min_val = torch.min(X)
    max_val = torch.max(X)
    epsilon = 1e-12      # small value to avoid division by zero
    normalized_X = (X - min_val) / (max_val - min_val + epsilon)
    return normalized_X

class VGG16Features(nn.Module):
    def __init__(self):
        super(VGG16Features, self).__init__()
        # Load the pre-trained VGG16 model
        vgg16_f = vgg16(pretrained=True).features

        # Define the blocks of VGG16 with their corresponding layers
        self.block1 = vgg16_f[:5]   # Conv2d_1-2 + ReLU + MaxPool2d
        self.block2 = vgg16_f[5:10] # Conv2d_2-2 + ReLU + MaxPool2d
        self.block3 = vgg16_f[10:17] # Conv2d_3-3 + ReLU + MaxPool2d
        self.block4 = vgg16_f[17:24] # Conv2d_4-3 + ReLU + MaxPool2d
        self.block5 = vgg16_f[24:31] # Conv2d_5-3 + ReLU + MaxPool2d
        
        # Freeze the parameters as we don't need to train them
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Pass the input through all blocks and store the intermediate outputs
        x = Resize([224,224])(x)
        x = normalize2(x)
        output1 = self.block1(x)
        output2 = self.block2(output1)
        output3 = self.block3(output2)
        output4 = self.block4(output3)
        output5 = self.block5(output4)
        
        # Return the outputs from each block
        return [output1, output2, output3, output4, output5]


class D_Loss(nn.Module):
    def __init__(self):
        super(D_Loss, self).__init__()
        self.beta = 0.35
        self.gamma = 0.35
        self.delta = 0.30
        self.tv = TotalVariation()
        #self.vgg = vgg16(pretrained=True)
        self.vgg16_features = VGG16Features().eval()
        

    def perceptual_sim_loss(self, prediction, label):
        # Extract VGG features and compute loss (example using one layer)
        prediction_features = self.vgg16_features(prediction)
        label_features = self.vgg16_features(label)
    
        # Compute the cosine similarity loss at each block
        #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #loss = sum([1 - F.cosine_similarity(a, b, dim=1) for a, b in zip(prediction_features, label_features)])
        c = []
        for a, b in zip(prediction_features, label_features):
            cos = (1 - F.cosine_similarity(a, b, dim=1)).mean()
            c.append(cos)
        loss = sum(c)
        return loss

    def structural_sim_loss(self, prediction, label):
    # SSIM loss is maximized when SSIM is maximized (1 - SSIM is minimized)
        loss = 1 - ssim(prediction, label, data_range=1, size_average=True)
        return loss

    def tv_loss(self, prediction):
        # Total variation loss
        N, C, H, W = prediction.shape
        #loss = torch.abs( (self.tv(prediction) / (N*C*H*W)) - 0.1 )
        loss = self.tv(prediction) / (N*C*H*W)
        return loss


    def forward(self, prediction, label):
        """
        """
        #l_sim = self.perceptual_similarity_loss(prediction, label)
        #l_vid = self.video_feature_loss(prediction, label)
        #l_r = self.regularization_term(prediction)
        #total_loss = self.beta * l_sim + self.gamma * l_vid + self.delta * l_r
        l_psim = self.perceptual_sim_loss(prediction, label)
        l_ssim = self.structural_sim_loss(prediction, label)
        l_tv = self.tv_loss(prediction)
        #l_gl = self.spatial_group_lasso(fc_weights)
        total_loss = self.beta * l_psim + self.gamma * l_ssim + self.delta * l_tv
        metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
        return l_psim.item(), l_ssim.item(), l_tv.item(), total_loss, metrics_names

### ENCODER-DECODER MODEL ###

class EncoderDecoder(nn.Module):
    def __init__(self, mask_size):
        super(EncoderDecoder, self).__init__()
        # Initialize the encoder and decoder components
        self.encoder = Encoder(mask_size)
        self.decoder = Decoder(mask_size)

    def forward(self, x):
        """
        Forward pass through both encoder and decoder.
        
        Parameters:
            x (tensor): Input to the encoder. Shape should be compatible with the Encoder's expected input shape.
        
        Returns:
            x (tensor): Output from the decoder. This would be the reconstructed version of the input if used in an autoencoder setup.
        """
        # Encode the input
        encoded = self.encoder(x)
        # Decode the encoded data
        decoded = self.decoder(encoded)
        return encoded, decoded


class ED_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Initializes the combined loss function for the EncoderDecoder model.
        It aggregates the loss from both the encoder and decoder.
        
        Args:
            alpha (float): Weighting factor for the encoder's cosine similarity term.
        """
        super(ED_Loss, self).__init__()
        # Initialize the encoder and decoder loss functions
        self.e_loss = E_Loss(alpha)
        self.d_loss = D_Loss()

    def forward(self, encoder_prediction, encoder_label, decoder_prediction, decoder_label):
        """
        Compute the combined loss for both encoder and decoder outputs.
        
        Args:
            encoder_prediction (torch.Tensor): Output from the encoder.
            encoder_label (torch.Tensor): Ground truth for the encoder output.
            decoder_prediction (torch.Tensor): Output from the decoder.
            decoder_label (torch.Tensor): Ground truth for the decoder output.
            fc_weights (torch.Tensor): Weights of the fully connected layer used in the loss computation of the decoder.
        
        Returns:
            tuple: Contains individual loss components and the total combined loss.
        """
        # Compute encoder loss
        _, e_loss, _ = self.e_loss(encoder_prediction, encoder_label)
        
        # Compute decoder loss
        _, _, _, d_loss, _ = self.d_loss(decoder_prediction, decoder_label)
        
        # Total combined loss is the sum of encoder and decoder losses
        total_loss = e_loss + d_loss

        metrics_names = ['encoder_loss', 'decoder_loss', 'combined_loss']
        
        return e_loss.item(), d_loss.item(), total_loss, metrics_names


### Training loop ###

def train_model(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None):
    """
    Universal training loop for models with variable output losses. Handles different models with a common interface.

    Parameters:
        input (array): Inputs of the trainset. Shape: (trainset_size, 3, 112, 112, 32).
        label (array): Labels of the trainset. Shape: (trainset_size, output_shape).
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (string): Device ('cpu' or 'cuda').
        save_model_as (string): Filename to save the trained model.
        start_epoch (int): Epoch to start/resume training.
        start_loss (float): Loss to start/resume training.
    Returns:
        model (torch.nn.Module): Trained model.
    """
    tic = time.time()

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_train not in model_type:
        print(f'model_to_train: {model_to_train} not recognized. Must be one of {model_type}')
        return None, None

    print(f'### Training {model_to_train} on input of shape {input.shape} ###')
    if pretrained_decoder:
        decoder = Decoder(label.shape[1])
        state_dict = torch.load(pretrained_decoder)
        decoder.load_state_dict(state_dict)
        decoder = decoder.to(device)
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        print(f'Also using pretrained decoder {pretrained_decoder}')

    print(f"Start training from epoch {start_epoch} with initial loss {start_loss}")
    
    input = torch.from_numpy(input)
    label = torch.from_numpy(label)

    train_set = torch.utils.data.TensorDataset(input, label)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=4,
    )
    
    model = model.to(device)
    criterion = criterion.to(device)

    history = {
        'total_loss': [],
        'other_metrics': [],
        'metrics_names': []
    }

    for epoch in range(start_epoch, num_epochs+1):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                model_outputs = model(inputs.float())
                if pretrained_decoder:
                    # If there's a pretrained decoder, use it with encoder outputs
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    # If no pretrained decoder, proceed with encoder outputs as main outputs
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels[..., 15])
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        if epoch % 5 == 0:  # Every 5 epochs, print status
            print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss / (5*i):.4f}")
            running_loss = 0.0
            save_checkpoint(model, optimizer, epoch+1, total_loss)

        history['total_loss'].append(total_loss.item())
        history['other_metrics'].append(loss_metrics)  # Store other metrics for visualization

    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    plot_train_losses(history, start_epoch)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history

def save_checkpoint(model, optimizer, epoch, loss):
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
    """
    filepath = 'checkpoint.pth'
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, device):
    """
    Load model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to load checkpoint parameters into.
        optimizer (torch.optim.Optimizer): Optimizer to load checkpoint parameters into.
    """

    model = model.to(device)

    filepath = 'checkpoint.pth'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def plot_train_losses(history, start_epoch):

    x = np.arange(len(history['total_loss'])) + start_epoch 
    
    plt.plot(x, history['total_loss'])
    plt.title(history['metrics_names'][-1])
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.show()

    for i in range(len(history['metrics_names']) - 1):
        plt.plot(x, history['other_metrics'][:, i])
        plt.title(history['metrics_names'][i])
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.show()

### Testing lopp ###

def test_model(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None):
    """
    Modular testing function for models with variable architectures (Encoder, Decoder, Encoder-Decoder).
    
    Parameters:
        inputs_dict (dict): Dictionary of input data arrays.
        labels_dict (dict): Dictionary of label data arrays.
        model (torch.nn.Module): Model to be tested.
        criterion (callable): Loss function that supports different configurations.
        device (str): Device to perform the testing on ('cpu' or 'cuda').
        pretrained_decoder (str, optional): Path to pretrained decoder model if used in an encoder-decoder setup.
        
    Returns:
        dict: A dictionary containing model predictions and loss metrics.
    """
    print('Start testing:')
    tic = time.time()

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_test not in model_type:
        print(f'model_to_test: {model_to_test} not recognized. Must be one of {model_type}')
        return None, None

    videos = list(inputs_dict.keys())
    inputs_shape = list(inputs_dict[videos[0]].shape)
    inputs_shape[0] = 'TR'
    print(f'### Testing {model_to_test} on inputs of shape {inputs_shape} over {len(videos)} videos ###')

    criterion = criterion.to(device)
    # Set model in testing phase
    model.to(device)
    model.eval()

    # Load and set pretrained decoder if specified
    #decoder = None
    if pretrained_decoder:
        decoder = Decoder(labels_dict[next(iter(labels_dict))].shape[1])  # Assuming shape is consistent across labels
        state_dict = torch.load(pretrained_decoder)
        decoder.load_state_dict(state_dict)
        decoder.to(device)
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        print(f'Also using pretrained decoder {pretrained_decoder}')

    if model_to_test != 'encoder_decoder' and pretrained_decoder is None:
        results = {
            model_to_test + '_predictions': {},
            'total_losses': {}
        }
    else:
        results = {
            'encoder_predictions': {},
            'decoder_predictions': {},
            'total_losses': {}
        }

    # Process each item in the inputs and labels dictionaries
    for key in inputs_dict.keys():
        input_tensor = torch.from_numpy(inputs_dict[key].astype('float32'))
        label_tensor = torch.from_numpy(labels_dict[key].astype('float32'))

        test_set = torch.utils.data.TensorDataset(input_tensor, label_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,  # Processing one TR at a time
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=4
        )

        model_outputs, decoder_outputs, total_losses = [], [], []
        with torch.no_grad():
            for input, label in test_loader:
                input, label = input.to(device), label.to(device)

                decoder_output = None
                
                if model_to_test == 'encoder_decoder':
                    model_output, decoder_output = model(input.float())
                elif pretrained_decoder:
                    model_output = model(input.float()).to(device)
                    decoder_output = decoder(model_output.float())
                else:
                    model_output = model(input.float())
                        
                model_outputs.append(model_output.detach().cpu())
                if decoder_output is not None:
                    decoder_outputs.append(decoder_output.detach().cpu())
            
                # Apply the appropriate criterion based on the presence of decoder outputs
                if model_to_test == 'decoder':
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label[..., 15])
                elif decoder_output is None:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label)
                else:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label, decoder_output, input[..., 15])
                
                total_losses.append(total_loss.item())

        if model_to_test != 'encoder_decoder' and pretrained_decoder is None:
            results[model_to_test + '_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
        else:
            results['encoder_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
            results['decoder_predictions'][key] = torch.cat(decoder_outputs, dim=0).numpy()
        
        results['total_losses'][key] = np.asarray(total_losses)

    print("Testing completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return results

