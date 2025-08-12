# -*- coding: utf-8 -*-

"""
This file contains functions related to model architectures, training, and testing.
The model architectures were inspired by the work of Kupershmidt et al. (2022). Project page: https://www.wisdom.weizmann.ac.il/~vision/VideoReconstFromFMRI/)

Models:
    Encoder
    Decoder
    EncoderDecoder

Functions:
    train_model
    save_checkpoint
    load_checkpoint
    test_model
"""

from imports import os, np, time, torch, nn, F, ssim, Resize, TotalVariation, vgg16
from dataset import normalize
from visualisation_new_2 import *
from dataset_new import load_durations


### ENCODER MODEL ###


class Encoder(nn.Module):
    def __init__(self, mask_size):
        """
        Initialize the Encoder architecture.
        
        Arguments:
            mask_size (int): The size of the output layer, corresponding to the size of the fMRI mask in use.
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
        Define the forward pass of the Encoder.
        
        Arguments:
            x (Tensor): The input data to the Encoder (shape of (TR, 3, 112, 112, 32)).
        
        Returns:
            Tensor: The encoded output (shape of (TR, mask_size)).
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

class E_Loss(nn.Module):
    def __init__(self, alpha=0.5, use_pretrained_decoder=False):
        """
        Initialize the encoder loss function.
        
        Arguments:
            alpha (float): Weight of the cosine similarity in the loss computation.
            use_pretrained_decoder (bool): Flag to determine if a pretrained decoder's loss should be included.
        """
        super(E_Loss, self).__init__()
        self.alpha = alpha
        self.use_pretrained_decoder = use_pretrained_decoder
        if self.use_pretrained_decoder:
            self.d_loss = D_Loss()  # Assuming D_Loss is defined elsewhere

    def forward(self, encoder_prediction, encoder_label, decoder_prediction=None, decoder_label=None):
        """
        Calculate the encoder loss and optionally the decoder loss if a pretrained decoder is used.
        
        Arguments:
            encoder_prediction (Tensor): Predictions from the encoder.
            encoder_label (Tensor): Ground truth labels for the encoder predictions.
            decoder_prediction (Tensor, optional): Predictions from the decoder, required if use_pretrained_decoder is True.
            decoder_label (Tensor, optional): Ground truth labels for the decoder predictions, required if use_pretrained_decoder is True.
        
        Returns:
            tuple: Contains loss values and metric names, structured based on whether decoder loss is included.
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
        """
        Initialize the Decoder architecture.
        
        Arguments:
            mask_size (int): The size of the input layer, corresponding to the size of the fMRI mask in use.
        """
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
        """
        Define the forward pass of the Decoder.
        
        Arguments:
            x (Tensor): The input data to the Decoder (shape of (TR, mask_size)).
        
        Returns:
            Tensor: The decoded output (shape of (TR, 3, 112, 112, 32)).
        """
    #    print("\n\nx shape at entry of forward =", x.shape, "\n\n")
        #print(x.shape)
        
        # Fully connected layer
        # x shape: (batch_size, mask_size)
        x = self.fc(x)
        # x.shape after self.fc = torch.Size([16, 9408 = 14*14*48])
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

        #print("\n\nx.shape before returning =", x.shape, "\n\n")

        # shape [16, 3, 112, 112]
        
        return x



'''
class TemporalDecoder(nn.Module):
    def __init__(self, mask_size, n_time_steps=3):
        super(TemporalDecoder, self).__init__()
        
        self.n_time_steps = n_time_steps
        
        # Process each TR individually first with shared weights
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        # 3D convolutional layers (process all TRs together)
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')  # Don't upsample temporal dim
        self.bn1 = nn.BatchNorm3d(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn3 = nn.BatchNorm3d(48)
        
        # Final layer maintains temporal dimension
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn4 = nn.BatchNorm3d(3)
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, mask_size)
        batch_size, n_time_steps, mask_size = x.shape
        
        # Process each TR independently first
        x_reshaped = x.reshape(-1, mask_size)  # Combine batch and time dimensions
        x = self.fc(x_reshaped)
        x = F.relu(x)
        
        # Reshape to 3D volume with temporal dimension
        x = x.view(batch_size, n_time_steps, 48, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, 48, n_time_steps, 14, 14)
        
        # 3D convolution processing
        x = self.conv1(x)
        x = F.relu(x)
        x = self.up1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.up2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.up3(x)
        x = self.bn3(x)
        
        # Final layer - maintains temporal dimension with kernel_size=(1,5,5)
        x = self.conv4(x)  # Output: (batch_size, 3, n_time_steps, 112, 112)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        # Rearrange to put channels after temporal dimension
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, n_time_steps, 3, 112, 112)
        
        return x
''' 

class TemporalDecoder(nn.Module):
    def __init__(self, mask_size, n_time_steps=3):
        """
        Initialize the Temporal Decoder.
        
        Arguments:
            mask_size (int): The size of the input layer, corresponding to the size of the fMRI mask in use.
            n_time_steps (int): Number of consecutive TRs to process together.
        """
        super(TemporalDecoder, self).__init__()
        
        self.n_time_steps = n_time_steps
        
        # Process each TR individually first with shared weights
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        # 3D convolutional layers (process all TRs together)
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')  # Don't upsample temporal dim
        self.bn1 = nn.BatchNorm3d(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn3 = nn.BatchNorm3d(48)
        
        # Final layer maintains temporal dimension
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn4 = nn.BatchNorm3d(3)
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, mask_size)
        batch_size, n_time_steps, mask_size = x.shape
        
        # Process each TR independently first
        x_reshaped = x.reshape(-1, mask_size)  # Combine batch and time dimensions
        x = self.fc(x_reshaped)
        x = F.relu(x)
        
        # Reshape to 3D volume with temporal dimension
        x = x.view(batch_size, n_time_steps, 48, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, 48, n_time_steps, 14, 14)
        
        # 3D convolution processing
        x = self.conv1(x)
        x = F.relu(x)
        x = self.up1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.up2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.up3(x)
        x = self.bn3(x)
        
        # Final layer - maintains temporal dimension with kernel_size=(1,5,5)
        x = self.conv4(x)  # Output: (batch_size, 3, n_time_steps, 112, 112)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        # Rearrange to put channels after temporal dimension
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, n_time_steps, 3, 112, 112)
        
        return x


class TemporalDecoder_old(nn.Module):
    def __init__(self, mask_size, n_time_steps=3):
        super(TemporalDecoder_old, self).__init__()
        
        self.n_time_steps = n_time_steps
        self.mask_size = mask_size
        
        # Process each TR individually first with shared weights
        self.fc = nn.Linear(mask_size, 14*14*48)    #input of size mask_size, output of 48 channels/features with 14x14 data each
        
        # 3D convolutional layers (process all TRs together)
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))  #kernel with 3 time steps makes convolution look at time evolution of each 5x5 pixels
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')  # Don't upsample temporal dim, only spatial, becomes 28x28
        self.bn1 = nn.BatchNorm3d(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')  #56x56
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')  #112x112
        self.bn3 = nn.BatchNorm3d(48)
        
        # Final layer maintains temporal dimension
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))   #48 features reduced to 3 (RGB)
        self.bn4 = nn.BatchNorm3d(3)
        
    #maybe i should divide the data into groups of 3 TRs before this and only accept groups of time_steps TRs
    def forward(self, x):
#        print("\n\nx shape at entry of forward =", x.shape, "\n\n")

        # x shape = [16, 4609], same as with normal Decoder

        # shape [16, 4609]
        
        # Check input shape and handle both single TRs and sequences
        if len(x.shape) == 2:  # Single TR input: [batch_size, mask_size]
            batch_size, mask_size = x.shape
            
            # Create a dummy temporal dimension (not sure if i should do this or just only accept groups of time_steps TRs since thats how the network was trained)
            x = x.unsqueeze(1)  # [batch_size, 1, mask_size]
            x = x.expand(-1, self.n_time_steps, -1)  # [batch_size, n_time_steps, mask_size]
            #print("\n\nx.shape after expanding (before reshaping) =", x.shape, "\n\n")

            # x.shape after expanding (before reshaping) = torch.Size([16, 3, 4609]) 

        # shape [16, 3, 4609]

            #i dont think this is the right way to do it. probably we should split the 16 frames into groups of 3 instead of putting 3 times the same frame

            n_time_steps = self.n_time_steps
        else:  # Temporal sequence: [batch_size, n_time_steps, mask_size]
            batch_size, n_time_steps, mask_size = x.shape
        
        # Process each TR independently first
        
        x_reshaped = x.reshape(-1, mask_size)  # Combine batch and time dimensions

        #print("\n\nx.shape before self.fc =", x_reshaped.shape, "\n\n")
        # x.shape before self.fc = torch.Size([48, 4609]) 

        # shape [48, 4609]

        x = self.fc(x_reshaped)

        #print("\n\nx.shape after self.fc =", x.shape, "\n\n")
        # x.shape after self.fc = torch.Size([48, 9408])
        # 16 TRs * 3 time steps = 48 , doesn't help that it's the same number of chosen channels
        # original x.shape after self.fc = torch.Size([16, 9408 = 14*14*48]

        # shape [48, 9408]

        x = F.relu(x)
        #print("\n\nx.shape after relu =", x.shape, "\n\n")

        # shape [48, 9408]
        
        # Reshape to 3D volume with temporal dimension
        x = x.view(batch_size, n_time_steps, 48, 14, 14)
        #print("\n\nx.shape after view =", x.shape, "\n\n")

        # shape [16, 3, 48, 14, 14]

        x = x.permute(0, 2, 1, 3, 4)  # Shape: [batch_size, 48, n_time_steps, 14, 14]
        #print("\n\nx.shape after permute =", x.shape, "\n\n")

        # shape [16, 48, 3, 14, 14]
        
        # 3D convolution processing
        x = self.conv1(x)
        #print("\n\nx.shape after conv1 =", x.shape, "\n\n")
        #self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))  #kernel with 3 time steps makes convolution look at time evolution of each 5x5 pixels
 
        # shape [16, 48, 3, 14, 14]

        x = F.relu(x)
        #print("\n\nx.shape after second relu =", x.shape, "\n\n")

        # shape [16, 48, 3, 14, 14]

        x = self.up1(x)
        #print("\n\nx.shape after up1 =", x.shape, "\n\n")

        # shape [16, 48, 3, 28, 28]

        x = self.bn1(x)
        #print("\n\nx.shape after bn1 =", x.shape, "\n\n")

        # shape [16, 48, 3, 28, 28]
        
        x = self.conv2(x)
        #print("\n\nx.shape after conv2 =", x.shape, "\n\n")

        # shape [16, 48, 3, 28, 28]

        x = F.relu(x)
        #print("\n\nx.shape after third relu =", x.shape, "\n\n")

        # shape [16, 48, 3, 28, 28]

        x = self.up2(x)
        #print("\n\nx.shape after up1 =", x.shape, "\n\n")

        # shape [16, 48, 3, 56, 56]

        x = self.bn2(x)
        #print("\n\nx.shape after bn2 =", x.shape, "\n\n")

        # shape [16, 48, 3, 56, 56]
        
        x = self.conv3(x)
        #print("\n\nx.shape after conv2 =", x.shape, "\n\n")

        # shape [16, 48, 3, 56, 56]

        x = F.relu(x)
        #print("\n\nx.shape after fourth relu =", x.shape, "\n\n")

        # shape [16, 48, 3, 56, 56]

        x = self.up3(x)
        #print("\n\nx.shape after up3 =", x.shape, "\n\n")

        # shape [16, 48, 3, 112, 112]

        x = self.bn3(x)
        #print("\n\nx.shape after bn3 =", x.shape, "\n\n")

        # shape [16, 48, 3, 112, 112]
        
        # Final layer - maintains temporal dimension with kernel_size=(1,5,5)
        x = self.conv4(x)  # Output: [batch_size, 3, n_time_steps, 112, 112]
        #print("\n\nx.shape after conv4 =", x.shape, "\n\n")

        # shape [16, 3, 3, 112, 112]

        x = torch.sigmoid(x)
        #print("\n\nx.shape after sigmoid =", x.shape, "\n\n")

        # shape [16, 3, 3, 112, 112]

        x = self.bn4(x)
        #print("\n\nx.shape after bn4 =", x.shape, "\n\n")

        # shape [16, 3, 3, 112, 112]
        
        # If input was a single TR, return only the middle frame (the frame that was repeated for n_time_steps)
        if len(x.shape) == 5:  # [batch_size, 3, n_time_steps, 112, 112]
            
            middle_idx = n_time_steps // 2
            x = x[:, :, middle_idx, :, :]  # [batch_size, 3, 112, 112]
            print("\nindeed the thing for middle frame ran\n")
        else:
            print("\nthe thing for middle frame didnt run\n")
        
        time.sleep(10)
        #print("\n\nx.shape after if len then middle frame =", x.shape, "\n\n")

        # shape [16, 3, 112, 112]
        # same as original forward ([16, 3, 112, 112])
        
        return x




class VGG16Features(nn.Module):
    def __init__(self):
        """
        Initialize the VGG16Features class to extract features using the VGG16 model pre-trained on ImageNet.
        """
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
        """
        Pass the input through the VGG16 blocks and return the intermediate outputs.
        
        Arguments:
            x (Tensor): The input tensor.
        
        Returns:
            list: A list of outputs from each block of the VGG16 model.
        """
        # Pass the input through all blocks and store the intermediate outputs

        #print("\n\nx.shape as input to vgg =", x.shape, "\n\n")

#        time.sleep(10)
        # original shape [16, 3, 112, 112]
        # temporal shape [16, 112, 112]
        # training temporal shape [16, 3, 112, 112]
        # new (after preparing data) (testing) temporal shape [16, 3, 112, 112]
        # wait actually thats only on the first x , second one is [16, 3, 3, 112, 112] ?

        x = Resize([224,224])(x)
        x = normalize(x)
        
#        print("x.shape =", x.shape)

#        time.sleep(1)

        #print("x =", x)
        
        output1 = self.block1(x)
        output2 = self.block2(output1)
        output3 = self.block3(output2)
        output4 = self.block4(output3)
        output5 = self.block5(output4)

        #print("output4.shape =", output4.shape)
        
        # Return the outputs from each block
        return [output1, output2, output3, output4, output5]


class D_Loss(nn.Module):
    def __init__(self):
        """
        Initialize the decoder loss function.
        """
        super(D_Loss, self).__init__()
        self.beta = 0.35
        self.gamma = 0.35
        self.delta = 0.30
#        self.beta = 0.6        #weight of psim loss on total loss
#        self.gamma = 0.25       #weight of ssim loss on total loss
#        self.delta = 0.15       #weight of tv loss on total loss
        #total_loss = self.beta * l_psim + self.gamma * l_ssim + self.delta * l_tv
        self.tv = TotalVariation()
        self.vgg16_features = VGG16Features().eval()
        

    def perceptual_sim_loss(self, prediction, label):
        # Extract VGG features and compute loss (example using one layer)
        prediction_features = self.vgg16_features(prediction)
        label_features = self.vgg16_features(label)
    
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
        loss = self.tv(prediction) / (N*C*H*W)
        return loss


    def forward(self, prediction, label):
        """
        Compute the total loss for the decoder predictions.
        
        Arguments:
            prediction (Tensor): Decoder predictions.
            label (Tensor): Ground truth labels.
        
        Returns:
            tuple: Contains individual loss components and the total loss.
        """
        l_psim = self.perceptual_sim_loss(prediction, label)
        l_ssim = self.structural_sim_loss(prediction, label)
        l_tv = self.tv_loss(prediction)
        total_loss = self.beta * l_psim + self.gamma * l_ssim + self.delta * l_tv
        metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
        return l_psim.item(), l_ssim.item(), l_tv.item(), total_loss, metrics_names


'''
class Temporal_D_Loss(nn.Module):
    def __init__(self):
        super(Temporal_D_Loss, self).__init__()
        # Keep your existing loss components
        self.beta = 0.35
        self.gamma = 0.35
        self.delta = 0.30
        self.tv = TotalVariation()
        self.vgg16_features = VGG16Features().eval()
        
    def forward(self, prediction, label):
        """
        Compute loss across all frames in the temporal sequence
        
        Arguments:
            prediction (Tensor): Shape (batch_size, n_time_steps, 3, 112, 112)
            label (Tensor): Shape (batch_size, n_time_steps, 3, 112, 112)
        """
        batch_size, n_time_steps = prediction.shape[0], prediction.shape[1]
        
        # Initialize loss components
        total_psim, total_ssim, total_tv = 0, 0, 0
        
        # Calculate loss for each frame in the sequence
        for t in range(n_time_steps):
            pred_frame = prediction[:, t]  # Shape: (batch_size, 3, 112, 112)
            label_frame = label[:, t]
            
            # Apply existing loss functions to each frame
            psim = self.perceptual_sim_loss(pred_frame, label_frame)
            ssim = self.structural_sim_loss(pred_frame, label_frame)
            tv = self.tv_loss(pred_frame)
            
            # Accumulate losses
            total_psim += psim
            total_ssim += ssim
            total_tv += tv
        
        # Average across temporal dimension
        total_psim /= n_time_steps
        total_ssim /= n_time_steps
        total_tv /= n_time_steps
        
        # Compute final loss with same weights
        total_loss = self.beta * total_psim + self.gamma * total_ssim + self.delta * total_tv
        
        metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
        return total_psim.item(), total_ssim.item(), total_tv.item(), total_loss, metrics_names
'''


class Temporal_D_Loss(nn.Module):
    def __init__(self):
        super(Temporal_D_Loss, self).__init__()
        # Keep your existing loss components
        self.beta = 0.35
        self.gamma = 0.35
        self.delta = 0.30
        self.tv = TotalVariation()
        self.vgg16_features = VGG16Features().eval()
        
    def perceptual_sim_loss(self, prediction, label):
        # Extract VGG features and compute loss (example using one layer)
        prediction_features = self.vgg16_features(prediction)
        label_features = self.vgg16_features(label)
    
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
        loss = self.tv(prediction) / (N*C*H*W)
        return loss
        
    def forward(self, prediction, label):
        """
        Compute loss across all frames in the temporal sequence
        
        Arguments:
            prediction (Tensor): Shape (batch_size, n_time_steps, 3, 112, 112)
            label (Tensor): Shape (batch_size, n_time_steps, 3, 112, 112)
        """
        batch_size, n_time_steps = prediction.shape[0], prediction.shape[1]
        
        # Initialize loss components
        total_psim, total_ssim, total_tv = 0, 0, 0
        
        # Calculate loss for each frame in the sequence
        for t in range(n_time_steps):
            pred_frame = prediction[:, t]  # Shape: (batch_size, 3, 112, 112)
            label_frame = label[:, t]
            
            # Apply existing loss functions to each frame
            psim = self.perceptual_sim_loss(pred_frame, label_frame)
            ssim = self.structural_sim_loss(pred_frame, label_frame)
            tv = self.tv_loss(pred_frame)
            
            # Accumulate losses
            total_psim += psim
            total_ssim += ssim
            total_tv += tv
        
        # Average across temporal dimension
        total_psim /= n_time_steps
        total_ssim /= n_time_steps
        total_tv /= n_time_steps
        
        # Compute final loss with same weights
        total_loss = self.beta * total_psim + self.gamma * total_ssim + self.delta * total_tv
        
        metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
        return total_psim.item(), total_ssim.item(), total_tv.item(), total_loss, metrics_names
    



class Temporal_D_Loss_old(nn.Module):
    def __init__(self):
        """
        Initialize the temporal decoder loss function.
        """

        print("it went in init")

        super(Temporal_D_Loss_old, self).__init__()
        self.beta = 0.35
        self.gamma = 0.35
        self.delta = 0.30
        self.tv = TotalVariation()
        self.vgg16_features = VGG16Features().eval()
        
    def perceptual_sim_loss(self, prediction, label):
        """
        Compute perceptual similarity loss using VGG features.
        """

        print("it went in perceptual_sim_loss")
        # Extract VGG features and compute loss (example using one layer)
        prediction_features = self.vgg16_features(prediction)
        label_features = self.vgg16_features(label)
    
        c = []
        for a, b in zip(prediction_features, label_features):
            cos = (1 - F.cosine_similarity(a, b, dim=1)).mean()
            c.append(cos)
        loss = sum(c)
        return loss

    def structural_sim_loss(self, prediction, label):
        """
        Compute structural similarity loss.
        """
        print("it went in structural_sim_loss")
        
        # SSIM loss is maximized when SSIM is maximized (1 - SSIM is minimized)
        loss = 1 - ssim(prediction, label, data_range=1, size_average=True)
        return loss

    def tv_loss(self, prediction):
        """
        Compute total variation loss.
        """
        print("it went in tv_loss")
        # Total variation loss
        N, C, H, W = prediction.shape
        loss = self.tv(prediction) / (N*C*H*W)
        return loss
        
    def forward(self, prediction, label):
        """
        Compute loss across all frames in the temporal sequence
        
        Arguments:
            prediction (Tensor): Shape (batch_size, n_time_steps, 3, 112, 112)
            label (Tensor): Shape (batch_size, n_time_steps, 3, 112, 112)
        
        Returns:
            tuple: Contains individual loss components and the total loss.
        """
#        print("it went in forward")
        l_psim = self.perceptual_sim_loss(prediction, label)
        l_ssim = self.structural_sim_loss(prediction, label)
        l_tv = self.tv_loss(prediction)
        total_loss = self.beta * l_psim + self.gamma * l_ssim + self.delta * l_tv
        metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
        
        return l_psim.item(), l_ssim.item(), l_tv.item(), total_loss, metrics_names

        batch_size, n_time_steps = prediction.shape[0], prediction.shape[1]
        
        # Initialize loss components
        total_psim, total_ssim, total_tv = 0, 0, 0
        
        # Calculate loss for each frame in the sequence
        for t in range(n_time_steps):
            pred_frame = prediction[:, t]  # Shape: (batch_size, 3, 112, 112)
            label_frame = label[:, t]
            
            # Apply existing loss functions to each frame
            print("pred_frame.shape, label_frame.shape =", pred_frame.shape, label_frame.shape)
            psim = self.perceptual_sim_loss(pred_frame, label_frame)
            ssim = self.structural_sim_loss(pred_frame, label_frame)
            tv = self.tv_loss(pred_frame)
            
            # Accumulate losses
            total_psim += psim
            total_ssim += ssim
            total_tv += tv
        
        # Average across temporal dimension
        total_psim /= n_time_steps
        total_ssim /= n_time_steps
        total_tv /= n_time_steps
        
        # Compute final loss with same weights
        total_loss = self.beta * total_psim + self.gamma * total_ssim + self.delta * total_tv
        
        metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
        print("it did the forward pass on temporal_d_loss")
        return total_psim.item(), total_ssim.item(), total_tv.item(), total_loss, metrics_names


def prepare_temporal_data(fmri_data, all_frames, window_size=3):
    """
    Prepare temporal data with overlapping windows of TRs and their corresponding middle frames
    
    Args:
        fmri_data: Original fMRI data, shape (n_trs, mask_size)
        all_frames: All video frames, organized by TR, shape (n_trs, 32, 3, height, width)
        window_size: Number of consecutive TRs to use
    
    Returns:
        tr_windows: Windows of consecutive TRs
        frame_targets: Middle frames for each TR in the windows
    """
    n_trs = fmri_data.shape[0]
    tr_windows = []
    frame_targets = []
    
    # Create sliding windows of TRs
    for i in range(n_trs - window_size + 1):
        # Get window of TRs
        tr_window = fmri_data[i:i+window_size]
        tr_windows.append(tr_window)
        
        # Get middle frame for each TR in the window
        frames_for_window = []
        for j in range(window_size):
            # Get middle frame (assuming 32 frames per TR)
            middle_frame_idx = 16  # Middle of 32 frames
            middle_frame = all_frames[i+j, middle_frame_idx]
            frames_for_window.append(middle_frame)
        
        frame_targets.append(torch.stack(frames_for_window))
    
    return torch.stack(tr_windows), torch.stack(frame_targets)



### ENCODER-DECODER MODEL ###


class EncoderDecoder(nn.Module):
    def __init__(self, mask_size):
        """
        Initialize the end-to-end Encoder-Decoder architecture.
        
        Arguments:
            mask_size (int): The size of the fMRI mask in use.
        """
        super(EncoderDecoder, self).__init__()
        # Initialize the encoder and decoder components
        self.encoder = Encoder(mask_size)
        self.decoder = Decoder(mask_size)

    def forward(self, x):
        """
        Define the forward pass through the end-to-end Encoder-Decoder.
        
        Arguments:
            x (Tensor): The input data to the Encoder-Decoder (shape of (TR, 3, 112, 112, 32)).
        
        Returns:
            tuple: Encoded and decoded outputs (respectively, shape of (TR, mask_size) and (TR, 3, 112, 112, 32)).
        """

        # Encode the input
        encoded = self.encoder(x)
        # Decode the encoded data
        decoded = self.decoder(encoded)
        return encoded, decoded


class ED_Loss(nn.Module):
    def __init__(self, encoder_weight=0.5, alpha=0.5):
        """
        Initialize the loss function for the Encoder-Decoder model.
        
        Arguments:
            encoder_weight (float): Weight of the encoder's loss in the combined loss computation.
            alpha (float): Parameter for the encoder's loss function (cosine distance weight).
        """
        super(ED_Loss, self).__init__()
        # Initialize the encoder and decoder loss functions
        self.encoder_weight = encoder_weight
        self.e_loss = E_Loss(alpha)
        self.d_loss = D_Loss()

    def forward(self, encoder_prediction, encoder_label, decoder_prediction, decoder_label):
        """
        Calculate and return the combined loss from the encoder and decoder outputs, along with individual loss components.
        
        Arguments:
            encoder_prediction (Tensor): Predictions from the encoder.
            encoder_label (Tensor): Actual labels for the encoder predictions.
            decoder_prediction (Tensor): Predictions from the decoder.
            decoder_label (Tensor): Actual labels for the decoder predictions.
        
        Returns:
            tuple: A tuple containing loss values and metric names, including individual and combined loss.
        """
        # Compute encoder loss
        _, e_loss, _ = self.e_loss(encoder_prediction, encoder_label)
        
        # Compute decoder loss
        _, _, _, d_loss, _ = self.d_loss(decoder_prediction, decoder_label)
        
        # Total combined loss is the sum of encoder and decoder losses
        total_loss = self.encoder_weight * e_loss + (1 - self.encoder_weight) * d_loss

        metrics_names = ['encoder_loss', 'decoder_loss', 'combined_loss']
        
        return e_loss.item(), d_loss.item(), total_loss, metrics_names


### TRAINING LOOP ###


#def train_model(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True):
def train_model(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name=""):
    """
    Train the model using the specified parameters and dataset.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.

    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training loss history.
    """
    tic = time.time()

    #print("input at beginning of function shape =", input.shape)
    #print("label at beginning of function shape =", label.shape)
    #time.sleep(10)

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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

    #print("\n\ntrain_loader shape =", train_loader.shape)
    #print("train_loader.dataset[0] shape =", train_loader.dataset[0].shape)
    #print("train_loader.dataset[0][1] shape =", train_loader.dataset[0][1].shape, "\n\n")
    #time.sleep(10)
    
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
        batch_count = 0
        epoch_metrics = None

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                model_outputs = model(inputs.float())       #???
                if pretrained_decoder:
                    # If there's a pretrained decoder, use it with encoder outputs
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    # If no pretrained decoder, proceed with encoder outputs as main outputs
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':

                #original training model_outputs shape [16, 3, 112, 112]
                #original labels[..., 15] shape [16, 3, 112, 112]
                #print("\n\nmodel_outputs shape =", model_outputs.shape)
                #print("labels[..., 15] shape =", labels[..., 15].shape, "\n\n")

                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels[..., 15])          #--> take the middle frame as label
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, torch.mean(labels, dim=4)) #--> take the average frame as label
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])          #--> middle frame
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, torch.mean(inputs, dim=4)) #--> average frame

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                for j, metric in enumerate(loss_metrics)]

            # Store metrics for averaging later
#            if epoch_metrics is None:
#                epoch_metrics = [metric.item() for metric in loss_metrics]
#            else:
#                epoch_metrics = [epoch_metrics[j] + metric.item() for j, metric in enumerate(loss_metrics)]

        # Calculate average loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
#            print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss / (5*i):.4f}")
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)
#            running_loss = 0.0
#            save_checkpoint(model, optimizer, epoch+1, total_loss)

#        history['total_loss'].append(total_loss.item())
#        history['other_metrics'].append(loss_metrics)  # Store other metrics for visualization

        history['total_loss'].append(avg_loss)  # Store average loss instead of just the last batch's loss
        history['other_metrics'].append(avg_metrics)  # Store average metrics for visualization

    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    if display_plots:
#        plot_train_losses(history, start_epoch)
        plot_train_losses(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history









def train_model_with_validation(input, label, val_input_dict, val_label_dict, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name=""):
    """
    Train the model using the specified parameters and dataset with validation.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        val_input_dict (dict): Dictionary of validation input data. Keys are movie names.
        val_label_dict (dict): Dictionary of validation labels. Keys are movie names.
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        display_plots (bool): Whether to display plots.
        save_plots (bool): Whether to save plots.
        model_name (str): Name for saving plots.

    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training and validation loss history.
    """
    tic = time.time()

    print("input at beginning of function shape =", input.shape)
    print("label at beginning of function shape =", label.shape)
    print("val_input_dict keys =", list(val_input_dict.keys()))
    print("val_label_dict keys =", list(val_label_dict.keys()))

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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
        'validation_loss': [],  # Add validation loss tracking
        'other_metrics': [],
        'metrics_names': []
    }
    
    for epoch in range(start_epoch, num_epochs+1):
        # Training phase
        model.train()                                   
        running_loss = 0.0
        batch_count = 0
        epoch_metrics = None

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                model_outputs = model(inputs.float())
                if pretrained_decoder:
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':
            #    print("model_outputs shape =", model_outputs.shape)
            #    print("labels[..., 15] shape =", labels[..., 15].shape)
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels[..., 15])
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # Clear some memory periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                for j, metric in enumerate(loss_metrics)]

        # Calculate average training loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        # Validation phase (run every 5 epochs)
        avg_val_loss = None
        if epoch % 5 == 0:
            model.eval()
            val_total_loss = 0.0
            val_total_batches = 0
            
            with torch.no_grad():
                # Process each key in validation dict (similar to temporal version)
                for key in val_input_dict.keys():
                    val_input_tensor = torch.from_numpy(val_input_dict[key].astype('float32'))
                    val_label_tensor = torch.from_numpy(val_label_dict[key].astype('float32'))
                    
                    val_set = torch.utils.data.TensorDataset(val_input_tensor, val_label_tensor)
                    val_loader = torch.utils.data.DataLoader(
                        val_set,
                        batch_size=8,  # Smaller batch size for validation
                        shuffle=False,
                        pin_memory=torch.cuda.is_available(),
                        num_workers=2
                    )
                    
                    for val_inputs, val_labels in val_loader:
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        
                        # Generate model outputs (same logic as training)
                        if model_to_train == 'encoder_decoder':
                            val_model_outputs, val_decoder_outputs = model(val_inputs.float())
                        else:
                            val_model_outputs = model(val_inputs.float())
                            if pretrained_decoder:
                                val_decoder_outputs = decoder(val_model_outputs.float())
                            else:
                                val_decoder_outputs = None
                        
                        # Apply criterion (same logic as training)
                        if model_to_train == 'decoder':
                            # Check if data has temporal dimension (5D vs 4D)
                            if len(val_labels.shape) == 5:  # Temporal data
                                *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels[..., 15])
                            else:  # Regular data
                                *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels)
                        elif val_decoder_outputs is None:
                            *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels)
                        else:
                            *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels, val_decoder_outputs, val_inputs[..., 15])
                        
                        val_total_loss += val_total_loss_batch.item()
                        val_total_batches += 1

            # Calculate average validation loss across all keys and batches
            avg_val_loss = val_total_loss / val_total_batches if val_total_batches > 0 else 0.0
            
            # Clear GPU memory after validation
            torch.cuda.empty_cache()

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
            if avg_val_loss is not None:
                print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)

        # Store losses in history
        history['total_loss'].append(avg_loss)
        # Only append validation loss when it's computed (every 5 epochs)
        if avg_val_loss is not None:
            history['validation_loss'].append(avg_val_loss)
        history['other_metrics'].append(avg_metrics)

    history['total_loss'] = np.asarray(history['total_loss'])
    history['validation_loss'] = np.asarray(history['validation_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    
    if display_plots:
        plot_train_losses_with_val(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history












'''def train_model_temporal(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name=""):
    """
    Train the model using the specified parameters and dataset.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.

    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training loss history.
    """
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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
        batch_count = 0
        epoch_metrics = None

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                model_outputs = model(inputs.float())       #???
                if pretrained_decoder:
                    # If there's a pretrained decoder, use it with encoder outputs
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    # If no pretrained decoder, proceed with encoder outputs as main outputs
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels[..., 15])          #--> take the middle frame as label
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, torch.mean(labels, dim=4)) #--> take the average frame as label
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])          #--> middle frame
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, torch.mean(inputs, dim=4)) #--> average frame

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                for j, metric in enumerate(loss_metrics)]

            # Store metrics for averaging later
#            if epoch_metrics is None:
#                epoch_metrics = [metric.item() for metric in loss_metrics]
#            else:
#                epoch_metrics = [epoch_metrics[j] + metric.item() for j, metric in enumerate(loss_metrics)]

        # Calculate average loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
#            print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss / (5*i):.4f}")
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)
#            running_loss = 0.0
#            save_checkpoint(model, optimizer, epoch+1, total_loss)

#        history['total_loss'].append(total_loss.item())
#        history['other_metrics'].append(loss_metrics)  # Store other metrics for visualization

        history['total_loss'].append(avg_loss)  # Store average loss instead of just the last batch's loss
        history['other_metrics'].append(avg_metrics)  # Store average metrics for visualization

    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    if display_plots:
#        plot_train_losses(history, start_epoch)
        plot_train_losses(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history'''






'''
def train_model_multi_subject(subjects, fMRIs_path, videos_path, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name=""):
    """
    Train the model using data from multiple subjects, loading data directly from video files.
    
    Arguments:
        subjects (list): List of subject IDs to use for training
        fMRIs_path (str): Path to fMRIs data (e.g., 'fMRIs_schaefer1000_4609')
        videos_path (str): Path to video data (e.g., 'processed_videos')
        model (nn.Module): The neural network model to be trained
        ... [other parameters same as original train_model]
    """
    tic = time.time()
    
    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)
    
    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_train not in model_type:
        print(f'model_to_train: {model_to_train} not recognized. Must be one of {model_type}')
        return None, None
    
    print(f'### Training {model_to_train} on data from {len(subjects)} subjects ###')
    if pretrained_decoder:
        decoder = Decoder(mask_size)  # You'll need to determine mask_size appropriately
        state_dict = torch.load(pretrained_decoder)
        decoder.load_state_dict(state_dict)
        decoder = decoder.to(device)
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        print(f'Also using pretrained decoder {pretrained_decoder}')
    
    print(f"Start training from epoch {start_epoch} with initial loss {start_loss}")
    
    # Get the list of available video files
    import glob
    video_files = [os.path.basename(f) for f in glob.glob(f"{videos_path}/*.npy")]
    
    if 'YouAgain.npy' in video_files:
        video_files.remove('YouAgain.npy')  # Remove test movie if present
    
    print(f"Found {len(video_files)} video files: {video_files}")
    
    # Print header for statistics
    print("\n===== SUBJECT DATA STATISTICS =====")
    print("Subject\tVideo\tInput Mean\tInput Var\tLabel Mean\tLabel Var")
    print("-" * 80)
    
    # Dictionary to store statistics
    subject_stats = {}
    
    # Initialize model for training
    model = model.to(device)
    criterion = criterion.to(device)
    
    history = {
        'total_loss': [],
        'other_metrics': [],
        'metrics_names': [],
        'epoch_details': []
    }
    
    # For each epoch
    for epoch in range(start_epoch, num_epochs+1):
        print(f"\nStarting Epoch {epoch}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        epoch_metrics = None
        batch_count = 0
        epoch_stats = {'subjects': {}}
        
        # For each subject
        for subject in subjects:
            subject_loss = 0.0
            subject_batch_count = 0
            subject_metrics = None
            epoch_stats['subjects'][subject] = {'videos': {}}
            
            print(f"Training on subject {subject}")
            
            # For each video
            for video_file in video_files:
                video_name = video_file[:-4]  # Remove .npy extension
                video_stats = {}
                
                try:
                    # Construct file paths
                    fmri_path = f"{fMRIs_path}/{subject}/{video_name}.npy"
                    video_path = f"{videos_path}/{video_name}.npy"
                    
                    # Check if files exist
                    if not os.path.exists(fmri_path):
                        print(f"Warning: {fmri_path} does not exist. Skipping.")
                        continue
                    
                    if not os.path.exists(video_path):
                        print(f"Warning: {video_path} does not exist. Skipping.")
                        continue
                    
                    # Load data
                    fmri_data = np.load(fmri_path)
                    video_data = np.load(video_path)
                    
                    # Calculate statistics
                    input_mean = np.mean(fmri_data)
                    input_var = np.var(fmri_data)
                    label_mean = np.mean(video_data)
                    label_var = np.var(video_data)
                    
                    # Store statistics
                    video_stats = {
                        'input_mean': input_mean,
                        'input_var': input_var,
                        'label_mean': label_mean,
                        'label_var': label_var,
                        'samples': fmri_data.shape[0]
                    }
                    
                    # Only print in the first epoch to avoid cluttering the output
                    if epoch == start_epoch:
                        print(f"{subject}\t{video_name}\t{input_mean:.6f}\t{input_var:.6f}\t{label_mean:.6f}\t{label_var:.6f}")
                    
                    # Training/validation split (80/20, excluding test portion)
                    TR = fmri_data.shape[0]
                    test_size = 0.2
                    test_sep = int(TR * (1 - test_size))
                    
                    # Use only the training portion (like in split_for_encoder)
                    train_indices = [i for i in range(test_sep) if i % 5 != 0]
                    
                    # Extract training data
                    train_input = fmri_data[train_indices]
                    train_label = video_data[train_indices]
                    
                    # Convert to tensors
                    train_input = torch.from_numpy(train_input).float()
                    train_label = torch.from_numpy(train_label).float()
                    
                    # Create data loader
                    train_dataset = torch.utils.data.TensorDataset(train_input, train_label)
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=torch.cuda.is_available(),
                        drop_last=False
                    )
                    
                    # Train on this video
                    video_loss = 0.0
                    video_batch_count = 0
                    video_metrics = None
                    
                    for i, (inputs, labels) in enumerate(train_loader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        
                        # Determine outputs based on model configuration
                        if model_to_train == 'encoder_decoder':
                            model_outputs, decoder_outputs = model(inputs.float())
                        else:
                            model_outputs = model(inputs.float())
                            if pretrained_decoder:
                                decoder_outputs = decoder(model_outputs.float())
                            else:
                                decoder_outputs = None
                        
                        # Apply appropriate criterion
                        if model_to_train == 'decoder':
                            *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels[..., 15])
                        elif decoder_outputs is None:
                            *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
                        else:
                            *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])
                        
                        total_loss.backward()
                        optimizer.step()
                        
                        loss_value = total_loss.item()
                        video_loss += loss_value
                        subject_loss += loss_value
                        epoch_loss += loss_value
                        
                        video_batch_count += 1
                        subject_batch_count += 1
                        batch_count += 1
                        
                        # Store metrics
                        current_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
                        
                        if video_metrics is None:
                            video_metrics = current_metrics
                        else:
                            video_metrics = [video_metrics[j] + current_metrics[j] for j in range(len(current_metrics))]
                            
                        if subject_metrics is None:
                            subject_metrics = current_metrics
                        else:
                            subject_metrics = [subject_metrics[j] + current_metrics[j] for j in range(len(current_metrics))]
                            
                        if epoch_metrics is None:
                            epoch_metrics = current_metrics
                        else:
                            epoch_metrics = [epoch_metrics[j] + current_metrics[j] for j in range(len(current_metrics))]
                    
                    # Store video metrics
                    if video_batch_count > 0:
                        avg_video_loss = video_loss / video_batch_count
                        avg_video_metrics = [metric / video_batch_count for metric in video_metrics] if video_metrics else []
                        
                        video_stats['loss'] = avg_video_loss
                        video_stats['metrics'] = avg_video_metrics
                        
                        print(f"  {video_name}: {video_batch_count} batches, Loss: {avg_video_loss:.4f}")
                    else:
                        print(f"  {video_name}: No batches trained")
                    
                    # Store video statistics
                    epoch_stats['subjects'][subject]['videos'][video_name] = video_stats
                    
                except Exception as e:
                    print(f"Error processing {subject}/{video_name}: {str(e)}")
            
            # Store subject metrics
            if subject_batch_count > 0:
                avg_subject_loss = subject_loss / subject_batch_count
                avg_subject_metrics = [metric / subject_batch_count for metric in subject_metrics] if subject_metrics else []
                
                epoch_stats['subjects'][subject]['loss'] = avg_subject_loss
                epoch_stats['subjects'][subject]['metrics'] = avg_subject_metrics
                
                print(f"Subject {subject}: {subject_batch_count} batches, Loss: {avg_subject_loss:.4f}")
            else:
                print(f"Subject {subject}: No batches trained")
        
        # Calculate epoch averages
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            avg_epoch_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []
            
            epoch_stats['loss'] = avg_epoch_loss
            epoch_stats['metrics'] = avg_epoch_metrics
            
            history['total_loss'].append(avg_epoch_loss)
            history['other_metrics'].append(avg_epoch_metrics)
            history['epoch_details'].append(epoch_stats)
            
            if epoch % 5 == 0 and display_plots:
                print(f"Epoch {epoch}: {batch_count} batches, Loss: {avg_epoch_loss:.4f}")
                save_checkpoint(model, optimizer, epoch+1, avg_epoch_loss)
        else:
            print(f"Epoch {epoch}: No batches trained")
    
    # Store metrics names
    if 'metrics_names' in locals():
        history['metrics_names'] = metrics_names
    
    # Save model
    torch.save(model.state_dict(), save_model_as)
    
    # Plot training losses if requested
    if display_plots and history['total_loss']:
        plot_train_losses(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)
    
    # Save history and statistics
    import pickle
    try:
        with open(f'training_history_{model_name}.pkl', 'wb') as f:
            pickle.dump(history, f)
        print(f"Saved training history to training_history_{model_name}.pkl")
    except Exception as e:
        print(f"Error saving history: {str(e)}")
    
    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history
'''




#just for thinking
def simple_train_model(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name=""):

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

    for epoch in range(start_epoch, num_epochs+1):
        model.train()                                   #???
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                model_outputs = model(inputs.float())       #???
                decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels[..., 15])          #--> take the middle frame as label
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, torch.mean(labels, dim=4)) #--> take the average frame as label
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])          #--> middle frame
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, torch.mean(inputs, dim=4)) #--> average frame

            total_loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_model_as)
    return model


def save_checkpoint(model, optimizer, epoch, loss):
    """
    Save the model checkpoint.

    Arguments:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
    """
    filepath = '/media/RCPNAS/MIP/Michael/students_work/rodrigo/checkpoint.pth'
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, device):
    """
    Load the model checkpoint.

    Arguments:
        model (nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        device (torch.device): Device to load the model onto (CPU or GPU).

    Returns:
        model (nn.Module): Loaded model.
        optimizer (torch.optim.Optimizer): Loaded optimizer.
        epoch (int): Last epoch number.
        loss (float): Last loss value.
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


### TESTING LOOP ###


def test_model(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None, statistical_testing=False, display_plots=True, save_plots=False, model_name="", temporal=False):
#def test_model(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None, statistical_testing = False, display_plots = True):
    """
    Test the pretrained model on the provided dataset.

    Arguments:
        inputs_dict (dict): Dictionary of input data. Keys are movie names. If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        labels_dict (dict): Dictionary of labels. Keys are movie names. If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The pretrained neural network model to be tested.
        criterion (nn.Module): Loss function for testing.
        device (torch.device): Device to test the model on (CPU or GPU).
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        model_to_test (str): Specifies which part of the model to test. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        statistical_testing (bool, optional): Whether to perform statistical testing. Default is False.

    Returns:
        results (dict): Dictionary containing test results including model predictions and losses.
    """
    print('Start testing:')
    tic = time.time()

    #print("labels_dict shape at the beginning: ")
    #labels_dict shape at the beginning (new labels, after using the prepare_temporal_data): 
    '''Sintel (shape: (109, 3, 3, 112, 112))
    Payload (shape: (153, 3, 3, 112, 112))
    TearsOfSteel (shape: (89, 3, 3, 112, 112))
    Superhero (shape: (156, 3, 3, 112, 112))
    BigBuckBunny (shape: (74, 3, 3, 112, 112))
    FirstBite (shape: (90, 3, 3, 112, 112))
    BetweenViewings (shape: (123, 3, 3, 112, 112))
    AfterTheRain (shape: (75, 3, 3, 112, 112))
    TheSecretNumber (shape: (119, 3, 3, 112, 112))
    Chatter (shape: (61, 3, 3, 112, 112))
    Spaceman (shape: (122, 3, 3, 112, 112))
    LessonLearned (shape: (101, 3, 3, 112, 112))
    YouAgain (shape: (611, 3, 3, 112, 112))
    ToClaireFromSonny (shape: (60, 3, 3, 112, 112))'''
    #print_dict_tree(labels_dict)
    #print("\n\n")

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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

        decoder_saliency = np.zeros(labels_dict[videos[0]].shape[1])

    results['test_performance'] = {}
    
    # Process each item in the inputs and labels dictionaries
    for key in inputs_dict.keys():
        input_tensor = torch.from_numpy(inputs_dict[key].astype('float32'))
        label_tensor = torch.from_numpy(labels_dict[key].astype('float32'))

        print("input_tensor shape before test_loader: ", label_tensor.shape)
        print("label_tensor shape before test_loader: ", label_tensor.shape)

        #input_tensor shape before test_loader:  torch.Size([109, 3, 3, 112, 112])
        #label_tensor shape before test_loader:  torch.Size([109, 3, 3, 112, 112])

        test_set = torch.utils.data.TensorDataset(input_tensor, label_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=16,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=4
        )


        model_outputs, decoder_outputs, total_losses = [], [], []
        #individual_metrics = {metric_name: [] for metric_name in ['perc_sim', 'struct_sim', 'tv_loss']}  # Use your actual metric names

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
                    print("input shape to generate model_output =", input.shape)
                    #input shape to generate model_output = torch.Size([16, 3, 4609]
                    #inputs shape to generate model_output = torch.Size([16, 3, 4609]
                    model_output = model(input.float())
                    print("model_output shape just after generating =", model_output.shape)
                    # new temporal model_output shape just after generating [16, 3, 112, 112]
                    # temporal training model_outputs shape just after generating [16, 3, 3, 112, 112]
                    # time.sleep(10)
                        
                model_outputs.append(model_output.detach().cpu())
                if decoder_output is not None:
                    decoder_outputs.append(decoder_output.detach().cpu())
            
                # Apply the appropriate criterion based on the presence of decoder outputs

                #tag test_model

                if model_to_test == 'decoder':
                    #print("\n\nmodel_output shape before going to criterion: ", model_output.shape)
                    #print("label shape before going to criterion: ", label.shape, "\n\n")
                    
                    #time.sleep(10)
                    #original model_output shape [16, 3, 112, 112]
                    #temporal model_output shape [16, 3, 112, 112] (same)
                    #original label[..., 15] shape [16, 3, 112, 112]
                    #temporal label[..., 15] shape [16, 3, 112, 112] (same)

                    #original training model_outputs shape [16, 3, 112, 112]
                    #temporal training model_outputs shape [16, 3, 3, 112, 112]
                    
                    #original training labels[..., 15] shape [16, 3, 112, 112]
                    #temporal training labels shape [16, 3, 3, 112, 112]) (note that in training we dont select just the middle frame because that was done when preparing the data)

                    #new temporal model_output shape [16, 3, 112, 112] <- why?
                    #new temporal label shape [16, 3, 3, 112, 112]

                    #new temporal model_output shape [16, 3, 112, 112]
                    #new temporal label[..., 15] shape [16, 3, 3, 112] <- label is just from labels_dict, so this is weird


                    #something is wrong with the training label shape.
                    #original input at beginning of function shape = (4321, 4609)
                    #temporal input at beginning of function shape = (4319, 3, 4609)
                    #original label at beginning of function shape = (4321, 3, 112, 112, 32)
                    #temporal label at beginning of function shape = (4319, 3, 3, 112, 112)
                    
                    if temporal:
                        *loss_metrics, total_loss, metrics_names = criterion(model_output, label)
                        #can't use these losses for plotting because the return is one value per batch of 16 frames, not one value per frame
                        #for i, name in enumerate(metrics_names[:len(loss_metrics)]):
                        #    if name in individual_metrics:
                        #        print("loss_metrics[i] =", loss_metrics[i])
                        #        individual_metrics[name].append(loss_metrics[i])
                            
                        #metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
                        #return total_psim.item(), total_ssim.item(), total_tv.item(), total_loss, metrics_names
                    else:
                        *loss_metrics, total_loss, metrics_names = criterion(model_output, label[..., 15])          #--> middle frame
                    #*loss_metrics, total_loss, metrics_names = criterion(model_output, torch.mean(label, dim=4)) #--> average frame
                elif decoder_output is None:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label)
                else:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label, decoder_output, input[..., 15])          #--> middle frame
                    #*loss_metrics, total_loss, metrics_names = criterion(model_output, label, decoder_output, torch.mean(input, dim=4)) #--> average frame
                
                total_losses.append(total_loss.item())

                


        if model_to_test != 'encoder_decoder' and pretrained_decoder is None:
            results[model_to_test + '_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
        else:
            results['encoder_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
            results['decoder_predictions'][key] = torch.cat(decoder_outputs, dim=0).numpy()
        
        results['total_losses'][key] = np.asarray(total_losses)
        #if 'individual_metrics' not in results:
        #    results['individual_metrics'] = {}
        #    results['individual_metrics'][key] = {metric: np.asarray(values) for metric, values in individual_metrics.items()}

        if model_to_test != 'decoder':
            encoded = results['encoder_predictions'][key]
            labels = labels_dict[key]
            plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)

            #plot_metrics(labels, encoded, key, plot_TR = False, performance_dict = None, display_plots=display_plots)

    if model_to_test != 'decoder':
        all_encoded = results['encoder_predictions']
        all_labels = labels_dict
        results['test_performance'] = plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)
#        plot_metrics(all_labels, all_encoded, 'all', plot_TR = True, performance_dict = results['test_performance'], display_plots = display_plots)
        if statistical_testing:
            all_labels, all_predictions = [], []
            for key in labels_dict.keys():
                all_predictions.append(all_predictions[key])
                all_labels.append(all_labels[key])
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            one_sample_permutation_test(all_labels, all_predictions)

    if model_to_test != 'encoder' or pretrained_decoder is not None:
        if model_to_test == 'decoder':
            results['test_performance'] = plot_decoder_predictions(results['decoder_predictions'], labels_dict, results['test_performance'], display_plots, save_plots=save_plots, save_path_prefix='outputs/' if save_plots else None, model_name=model_name, temporal=temporal, 
            #individual_metrics=results['individual_metrics']
            )

            #plot_decoder_predictions(
#                results['decoder_predictions'], labels_dict, 
#                results['test_performance'], display_plots,
#                save_plots=save_plots,
#                save_path_prefix='outputs/' if save_plots else None,
#                model_name=model_name)

#            plot_decoder_predictions(results['decoder_predictions'], labels_dict, results['test_performance'], display_plots)
        else:
            results['test_performance'] = plot_decoder_predictions(
                results['decoder_predictions'], inputs_dict, 
                results['test_performance'], display_plots,
                save_plots=save_plots,
                save_path_prefix='outputs/' if save_plots else None,
                model_name=model_name)
#            plot_decoder_predictions(results['decoder_predictions'], inputs_dict, results['test_performance'], display_plots)
        
    if model_to_test == 'encoder_decoder':
        with torch.enable_grad():
            for key in inputs_dict.keys():
                predicted_fMRIs = torch.from_numpy(results['encoder_predictions'][key])
                ground_truth_frames = torch.from_numpy(inputs_dict[key][..., 15])
                for i in range(predicted_fMRIs.shape[0]):
                    decoder_saliency += compute_saliency(model.decoder, predicted_fMRIs[i:i+1], ground_truth_frames[i:i+1], device)

        if display_plots:
            plot_saliency_distribution(decoder_saliency)
        results['decoder_saliency'] = decoder_saliency

    print("Testing completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return results




def test_model_all(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None, 
               statistical_testing=False, display_plots=True, save_plots=False, model_name="", metric="ssim", 
               mean_flag=False, zones=None, baseline_predictions=None):
    """
    Test the pretrained model on the provided dataset.

    Arguments:
        inputs_dict (dict): Dictionary of input data. Keys are movie names or slice identifiers. 
                           If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, 3, 112, 112, 32). 
                           Else, shape of (TR, mask_size).
        labels_dict (dict): Dictionary of labels. Keys are movie names. 
                           If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, mask_size). 
                           Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The pretrained neural network model to be tested.
        criterion (nn.Module): Loss function for testing.
        device (torch.device): Device to test the model on (CPU or GPU).
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        model_to_test (str): Specifies which part of the model to test. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        statistical_testing (bool, optional): Whether to perform statistical testing. Default is False.
        display_plots (bool, optional): Whether to display plots. Default is True.
        save_plots (bool, optional): Whether to save plots. Default is False.
        model_name (str, optional): Name of the model for saving plots. Default is "".
        zones (str or int, optional): Zones to consider for testing. Default is "quadrants", can also be "center_bg".
                                      If it is an integer, the function will analyze a number of zones = that integer squared.
                                      For example, if zones = 4, the function will analyze 16 zones (4x4).
        baseline_predictions (dict, optional): Dictionary of baseline predictions for comparison.

    Returns:
        results (dict): Dictionary containing test results including model predictions and losses.
    """
    print('Start testing:')
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_test not in model_type:
        print(f'model_to_test: {model_to_test} not recognized. Must be one of {model_type}')
        return None, None

    # Get list of input keys (movie names or slice identifiers)
    videos = list(inputs_dict.keys())
    inputs_shape = list(inputs_dict[videos[0]].shape)
    inputs_shape[0] = 'TR'
    print(f'### Testing {model_to_test} on inputs of shape {inputs_shape} over {len(videos)} videos/slices ###')
    
    if baseline_predictions is not None:
        print(f'### Using baseline predictions for comparison ###')

    criterion = criterion.to(device)
    # Set model in testing phase
    model.to(device)
    model.eval()

    # Load and set pretrained decoder if specified
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

        decoder_saliency = np.zeros(labels_dict[list(labels_dict.keys())[0]].shape[1])

    results['test_performance'] = {}
    
    # Process each item in the inputs and labels dictionaries
    for key in inputs_dict.keys():
        input_tensor = torch.from_numpy(inputs_dict[key].astype('float32'))
        
        # Get the corresponding label - if it's a slice name, extract the original movie name
        if key in labels_dict:
            label_key = key
        else:
            # Extract the movie name from the key (assumed to be after the last underscore)
            # For keys like "slice_0_Payload", this will extract "Payload"
            if '_' in key:
                extracted_movie = key.split('_')[-1]
                if extracted_movie in labels_dict:
                    label_key = extracted_movie
                    print(f"Input key '{key}' not found in labels. Extracted and using '{label_key}' for labels.")
                else:
                    # If extracted name not found, use first available label
                    label_key = list(labels_dict.keys())[0]
                    print(f"Input key '{key}' and extracted movie '{extracted_movie}' not found in labels. Using '{label_key}' for labels.")
            else:
                # If no underscore in key, use first available label
                label_key = list(labels_dict.keys())[0]
                print(f"Input key '{key}' not found in labels and no movie name could be extracted. Using '{label_key}' for labels.")
        
        label_tensor = torch.from_numpy(labels_dict[label_key].astype('float32'))

        # Debug info about tensors but without using print_dict_tree
        print(f"input_tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"label_tensor shape: {label_tensor.shape}, dtype: {label_tensor.dtype}")
        
        test_set = torch.utils.data.TensorDataset(input_tensor, label_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=16,
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
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label[..., 15])          #--> middle frame
                elif decoder_output is None:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label)
                else:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label, decoder_output, input[..., 15])
                
                total_losses.append(total_loss.item())

        # Store the outputs in results
        if model_to_test != 'encoder_decoder' and pretrained_decoder is None:
            results[model_to_test + '_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
        else:
            results['encoder_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
            results['decoder_predictions'][key] = torch.cat(decoder_outputs, dim=0).numpy()
        
        results['total_losses'][key] = np.asarray(total_losses)

        if model_to_test != 'decoder':
            encoded = results['encoder_predictions'][key]
            labels = labels_dict[label_key] if key not in labels_dict else labels_dict[key]
            plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)

    if model_to_test != 'decoder':
        all_encoded = results['encoder_predictions']
        all_labels = labels_dict
        # Using the last processed key for display
        results['test_performance'] = plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)

        if statistical_testing:
            all_labels, all_predictions = [], []
            for key in labels_dict.keys():
                if key in results['encoder_predictions']:
                    all_predictions.append(results['encoder_predictions'][key])
                    all_labels.append(labels_dict[key])
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            one_sample_permutation_test(all_labels, all_predictions)

    if model_to_test != 'encoder' or pretrained_decoder is not None:
        if model_to_test == 'decoder':
            print("\n\n\n ALRIGHT ZONES =", zones, "\n\n\n")
            
            # Check if we have baseline predictions to use
            if baseline_predictions is not None:
                # Use the modified plot_all_predictions7 with baseline comparison
                results['test_performance'] = plot_all_predictions7(
                    results['decoder_predictions'], 
                    labels_dict, 
                    results['test_performance'], 
                    display_plots,
                    save_plots=save_plots,
                    save_path_prefix='outputs/' if save_plots else None,
                    model_name=model_name, 
                    metric=metric, 
                    mean_flag=mean_flag,
                    zone_type=zones,
                    baseline_predictions=baseline_predictions
                )
            else:
                # Use the regular plot_all_predictions7 without baseline
                if zones is None:
                    results['test_performance'] = plot_all_predictions5(
                        results['decoder_predictions'], 
                        labels_dict, 
                        results['test_performance'], 
                        display_plots,
                        save_plots=save_plots,
                        save_path_prefix='outputs/' if save_plots else None,
                        model_name=model_name, 
                        metric=metric, 
                        mean_flag=mean_flag
                    )
                else:
                    results['test_performance'] = plot_all_predictions7(
                        results['decoder_predictions'], 
                        labels_dict, 
                        results['test_performance'], 
                        display_plots,
                        save_plots=save_plots,
                        save_path_prefix='outputs/' if save_plots else None,
                        model_name=model_name, 
                        metric=metric, 
                        mean_flag=mean_flag,
                        zone_type=zones
                    )

        else:
            # For encoder or encoder_decoder, use inputs_dict for ground truth
            if baseline_predictions is not None:
                results['test_performance'] = plot_all_predictions7(
                    results['decoder_predictions'], 
                    inputs_dict, 
                    results['test_performance'], 
                    display_plots,
                    save_plots=save_plots,
                    save_path_prefix='outputs/' if save_plots else None,
                    model_name=model_name, 
                    metric=metric, 
                    baseline_predictions=baseline_predictions
                )
            else:
                results['test_performance'] = plot_all_predictions5(
                    results['decoder_predictions'], 
                    inputs_dict, 
                    results['test_performance'], 
                    display_plots,
                    save_plots=save_plots,
                    save_path_prefix='outputs/' if save_plots else None,
                    model_name=model_name, 
                    metric=metric
                )
    print("using new function")
        
    if model_to_test == 'encoder_decoder':
        with torch.enable_grad():
            for key in inputs_dict.keys():
                predicted_fMRIs = torch.from_numpy(results['encoder_predictions'][key])
                # Get corresponding input for ground truth
                if key in inputs_dict:
                    input_key = key
                else:
                    # Use first input if key not found
                    input_key = list(inputs_dict.keys())[0]
                
                ground_truth_frames = torch.from_numpy(inputs_dict[input_key][..., 15])
                for i in range(predicted_fMRIs.shape[0]):
                    decoder_saliency += compute_saliency(model.decoder, predicted_fMRIs[i:i+1], ground_truth_frames[i:i+1], device)

        if display_plots:
            plot_saliency_distribution(decoder_saliency)
        results['decoder_saliency'] = decoder_saliency

    print("Testing completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return results





#this one was working ok
'''
def test_model_all(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None, statistical_testing=False, display_plots=True, save_plots=False, model_name="", metric="ssim", mean_flag=False, zones=None):
    """
    Test the pretrained model on the provided dataset.

    Arguments:
        inputs_dict (dict): Dictionary of input data. Keys are movie names or slice identifiers. 
                           If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, 3, 112, 112, 32). 
                           Else, shape of (TR, mask_size).
        labels_dict (dict): Dictionary of labels. Keys are movie names. 
                           If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, mask_size). 
                           Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The pretrained neural network model to be tested.
        criterion (nn.Module): Loss function for testing.
        device (torch.device): Device to test the model on (CPU or GPU).
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        model_to_test (str): Specifies which part of the model to test. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        statistical_testing (bool, optional): Whether to perform statistical testing. Default is False.
        display_plots (bool, optional): Whether to display plots. Default is True.
        save_plots (bool, optional): Whether to save plots. Default is False.
        model_name (str, optional): Name of the model for saving plots. Default is "".

        zones (str or int, optional): Zones to consider for testing. Default is "quadrants", can also be "center_bg".
                                      If it is an integer, the function will analyze a number of zones = that integer squared.
                                      For example, if zones = 4, the function will analyze 16 zones (4x4).

    Returns:
        results (dict): Dictionary containing test results including model predictions and losses.
    """
    print('Start testing:')
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_test not in model_type:
        print(f'model_to_test: {model_to_test} not recognized. Must be one of {model_type}')
        return None, None

    # Get list of input keys (movie names or slice identifiers)
    videos = list(inputs_dict.keys())
    inputs_shape = list(inputs_dict[videos[0]].shape)
    inputs_shape[0] = 'TR'
    print(f'### Testing {model_to_test} on inputs of shape {inputs_shape} over {len(videos)} videos/slices ###')

    criterion = criterion.to(device)
    # Set model in testing phase
    model.to(device)
    model.eval()

    # Load and set pretrained decoder if specified
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

        decoder_saliency = np.zeros(labels_dict[list(labels_dict.keys())[0]].shape[1])

    results['test_performance'] = {}
    
    # Process each item in the inputs and labels dictionaries
    for key in inputs_dict.keys():
        input_tensor = torch.from_numpy(inputs_dict[key].astype('float32'))
        
        # Get the corresponding label - if it's a slice name, extract the original movie name
        if key in labels_dict:
            label_key = key
        else:
            # Extract the movie name from the key (assumed to be after the last underscore)
            # For keys like "slice_0_Payload", this will extract "Payload"
            if '_' in key:
                extracted_movie = key.split('_')[-1]
                if extracted_movie in labels_dict:
                    label_key = extracted_movie
                    print(f"Input key '{key}' not found in labels. Extracted and using '{label_key}' for labels.")
                else:
                    # If extracted name not found, use first available label
                    label_key = list(labels_dict.keys())[0]
                    print(f"Input key '{key}' and extracted movie '{extracted_movie}' not found in labels. Using '{label_key}' for labels.")
            else:
                # If no underscore in key, use first available label
                label_key = list(labels_dict.keys())[0]
                print(f"Input key '{key}' not found in labels and no movie name could be extracted. Using '{label_key}' for labels.")
        
        label_tensor = torch.from_numpy(labels_dict[label_key].astype('float32'))

        # Debug info about tensors but without using print_dict_tree
        print(f"input_tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"label_tensor shape: {label_tensor.shape}, dtype: {label_tensor.dtype}")
        
        test_set = torch.utils.data.TensorDataset(input_tensor, label_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=16,
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
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label[..., 15])          #--> middle frame
                elif decoder_output is None:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label)
                else:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label, decoder_output, input[..., 15])
                
                total_losses.append(total_loss.item())

        # Store the outputs in results
        if model_to_test != 'encoder_decoder' and pretrained_decoder is None:
            results[model_to_test + '_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
        else:
            results['encoder_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
            results['decoder_predictions'][key] = torch.cat(decoder_outputs, dim=0).numpy()
        
        results['total_losses'][key] = np.asarray(total_losses)

        if model_to_test != 'decoder':
            encoded = results['encoder_predictions'][key]
            labels = labels_dict[label_key] if key not in labels_dict else labels_dict[key]
            plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)

    if model_to_test != 'decoder':
        all_encoded = results['encoder_predictions']
        all_labels = labels_dict
        # Using the last processed key for display
        results['test_performance'] = plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)

        if statistical_testing:
            all_labels, all_predictions = [], []
            for key in labels_dict.keys():
                if key in results['encoder_predictions']:
                    all_predictions.append(results['encoder_predictions'][key])
                    all_labels.append(labels_dict[key])
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            one_sample_permutation_test(all_labels, all_predictions)

    if model_to_test != 'encoder' or pretrained_decoder is not None:
        if model_to_test == 'decoder':
            print("\n\n\n ALRIGHT ZONES =", zones, "\n\n\n")
            # Pass all decoder predictions to plot_all_predictions
#            results['test_performance'] = plot_all_predictions4(
#                results['decoder_predictions'], labels_dict, 
#                results['test_performance'], display_plots,
#                save_plots=save_plots,
#                save_path_prefix='outputs/' if save_plots else None,
#                model_name=model_name)
            if zones is None:
                results['test_performance'] = plot_all_predictions5(
                    results['decoder_predictions'], labels_dict, 
                    results['test_performance'], display_plots,
                    save_plots=save_plots,
                    save_path_prefix='outputs/' if save_plots else None,
                    model_name=model_name, metric=metric, mean_flag=mean_flag)
            
            else:
                results['test_performance'] = plot_all_predictions7(
#                results['test_performance'] = plot_all_predictions6(
                    results['decoder_predictions'], labels_dict, 
                    results['test_performance'], display_plots,
                    save_plots=save_plots,
                    save_path_prefix='outputs/' if save_plots else None,
                    model_name=model_name, metric=metric, mean_flag=mean_flag,
                    zone_type=zones)

        else:
            # For encoder or encoder_decoder, use inputs_dict for ground truth
#            results['test_performance'] = plot_all_predictions4(
#                results['decoder_predictions'], inputs_dict, 
#                results['test_performance'], display_plots,
#                save_plots=save_plots,
#                save_path_prefix='outputs/' if save_plots else None,
#                model_name=model_name)
            results['test_performance'] = plot_all_predictions5(
                results['decoder_predictions'], inputs_dict, 
                results['test_performance'], display_plots,
                save_plots=save_plots,
                save_path_prefix='outputs/' if save_plots else None,
                model_name=model_name, metric=metric)
    print("using new function")
        
    if model_to_test == 'encoder_decoder':
        with torch.enable_grad():
            for key in inputs_dict.keys():
                predicted_fMRIs = torch.from_numpy(results['encoder_predictions'][key])
                # Get corresponding input for ground truth
                if key in inputs_dict:
                    input_key = key
                else:
                    # Use first input if key not found
                    input_key = list(inputs_dict.keys())[0]
                
                ground_truth_frames = torch.from_numpy(inputs_dict[input_key][..., 15])
                for i in range(predicted_fMRIs.shape[0]):
                    decoder_saliency += compute_saliency(model.decoder, predicted_fMRIs[i:i+1], ground_truth_frames[i:i+1], device)

        if display_plots:
            plot_saliency_distribution(decoder_saliency)
        results['decoder_saliency'] = decoder_saliency

    print("Testing completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return results
'''

'''def test_model_all(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None, statistical_testing=False, display_plots=True, save_plots=False, model_name=""):
#def test_model(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None, statistical_testing = False, display_plots = True):
    """
    Test the pretrained model on the provided dataset.

    Arguments:
        inputs_dict (dict): Dictionary of input data. Keys are movie names. If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        labels_dict (dict): Dictionary of labels. Keys are movie names. If model_to_test is 'encoder' or 'encoder_decoder', then elements have a shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The pretrained neural network model to be tested.
        criterion (nn.Module): Loss function for testing.
        device (torch.device): Device to test the model on (CPU or GPU).
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        model_to_test (str): Specifies which part of the model to test. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        statistical_testing (bool, optional): Whether to perform statistical testing. Default is False.

    Returns:
        results (dict): Dictionary containing test results including model predictions and losses.
    """
    print('Start testing:')
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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

        decoder_saliency = np.zeros(labels_dict[videos[0]].shape[1])

    results['test_performance'] = {}
    
    # Process each item in the inputs and labels dictionaries
    for key in inputs_dict.keys():
        input_tensor = torch.from_numpy(inputs_dict[key].astype('float32'))
        label_tensor = torch.from_numpy(labels_dict[key].astype('float32'))

        test_set = torch.utils.data.TensorDataset(input_tensor, label_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=16,
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
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label[..., 15])          #--> middle frame
                    #*loss_metrics, total_loss, metrics_names = criterion(model_output, torch.mean(label, dim=4)) #--> average frame
                elif decoder_output is None:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label)
                else:
                    *loss_metrics, total_loss, metrics_names = criterion(model_output, label, decoder_output, input[..., 15])          #--> middle frame
                    #*loss_metrics, total_loss, metrics_names = criterion(model_output, label, decoder_output, torch.mean(input, dim=4)) #--> average frame
                
                total_losses.append(total_loss.item())

                


        if model_to_test != 'encoder_decoder' and pretrained_decoder is None:
            results[model_to_test + '_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
        else:
            results['encoder_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
            results['decoder_predictions'][key] = torch.cat(decoder_outputs, dim=0).numpy()
        
        results['total_losses'][key] = np.asarray(total_losses)

        if model_to_test != 'decoder':
            encoded = results['encoder_predictions'][key]
            labels = labels_dict[key]
            plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)

            #plot_metrics(labels, encoded, key, plot_TR = False, performance_dict = None, display_plots=display_plots)

    if model_to_test != 'decoder':
        all_encoded = results['encoder_predictions']
        all_labels = labels_dict
        results['test_performance'] = plot_metrics(labels, encoded, key, plot_TR=False, performance_dict=None, 
                        display_plots=display_plots,
                        save_plots=save_plots,
                        save_path=f'outputs/{key}_{model_name}.png' if save_plots else None)
#        plot_metrics(all_labels, all_encoded, 'all', plot_TR = True, performance_dict = results['test_performance'], display_plots = display_plots)
        if statistical_testing:
            all_labels, all_predictions = [], []
            for key in labels_dict.keys():
                all_predictions.append(all_predictions[key])
                all_labels.append(all_labels[key])
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            one_sample_permutation_test(all_labels, all_predictions)

    if model_to_test != 'encoder' or pretrained_decoder is not None:
        if model_to_test == 'decoder':
            results['test_performance'] = plot_all_predictions(results['decoder_predictions'], labels_dict, results['test_performance'], display_plots, save_plots=save_plots, save_path_prefix='outputs/' if save_plots else None, model_name=model_name)

            #plot_decoder_predictions(
#                results['decoder_predictions'], labels_dict, 
#                results['test_performance'], display_plots,
#                save_plots=save_plots,
#                save_path_prefix='outputs/' if save_plots else None,
#                model_name=model_name)

#            plot_decoder_predictions(results['decoder_predictions'], labels_dict, results['test_performance'], display_plots)
        else:
            results['test_performance'] = plot_all_predictions(
                results['decoder_predictions'], inputs_dict, 
                results['test_performance'], display_plots,
                save_plots=save_plots,
                save_path_prefix='outputs/' if save_plots else None,
                model_name=model_name)
#            plot_decoder_predictions(results['decoder_predictions'], inputs_dict, results['test_performance'], display_plots)
        
    if model_to_test == 'encoder_decoder':
        with torch.enable_grad():
            for key in inputs_dict.keys():
                predicted_fMRIs = torch.from_numpy(results['encoder_predictions'][key])
                ground_truth_frames = torch.from_numpy(inputs_dict[key][..., 15])
                for i in range(predicted_fMRIs.shape[0]):
                    decoder_saliency += compute_saliency(model.decoder, predicted_fMRIs[i:i+1], ground_truth_frames[i:i+1], device)

        if display_plots:
            plot_saliency_distribution(decoder_saliency)
        results['decoder_saliency'] = decoder_saliency

    print("Testing completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return results

'''


#just for thinking

#used with
    #load decoder part of encoder decoder
 #   model = Decoder(mask_size)
    #save_model_as = 'decoder_4609_50'
    #save_model_as = 'decoder_4609_1650'
#    state_dict = torch.load(model_name)
#    model.load_state_dict(state_dict)
    #model = model1.decoder

    #load data
#    if real:
#        test_input = testset['fMRIs']
#        test_label = testset['videos']
#    else:
#        test_input = model1.encoder(testset['videos'])
#        test_label = testset['videos']

#    criterion = D_Loss()
#    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#    pretrained_decoder=None
#    model_to_test='decoder'
#    statistical_testing = False
#    display_plots = True
#    save_plots = True

#    test_model(test_input, test_label, model, criterion, device, pretrained_decoder, model_to_test, statistical_testing, display_plots, save_plots, model_name=model_name)
    

def simple_test_model(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None,
                      model_to_test=None, statistical_testing=False, display_plots=True, save_plots=False, model_name=""):
    tic = time.time()

    videos = list(inputs_dict.keys())
    inputs_shape = list(inputs_dict[videos[0]].shape)
    inputs_shape[0] = 'TR'
    
    criterion = criterion.to(device)
    # Set model in testing phase
    model.to(device)
    model.eval()

    # Load and set pretrained decoder if specified

    results = {
        'encoder_predictions': {},
        'decoder_predictions': {},
        'total_losses': {},
        'test_performance' : {}
    }
    
    # Process each item in the inputs and labels dictionaries
    # repeat this for each video
    for key in inputs_dict.keys():
        input_tensor = torch.from_numpy(inputs_dict[key].astype('float32')) #inputs_dict is testset['fMRIs']
        label_tensor = torch.from_numpy(labels_dict[key].astype('float32')) #labels_dict is testset['videos']

        test_set = torch.utils.data.TensorDataset(input_tensor, label_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=16,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=4
        )

        model_outputs, decoder_outputs, total_losses = [], [], []
        with torch.no_grad():
            for input, label in test_loader:    #for each batch i think, so for each 16 chunks of 32 frames
                input, label = input.to(device), label.to(device)
            
                decoder_output = None
                model_output = model(input.float())

                model_outputs.append(model_output.detach().cpu())
                if decoder_output is not None:
                    decoder_outputs.append(decoder_output.detach().cpu())

                # Apply the appropriate criterion based on the presence of decoder outputs
                if model_to_test == 'decoder':
                    *___, total_loss, __ = criterion(model_output, label[..., 15])          #--> middle frame
                
                total_losses.append(total_loss.item())


        results[model_to_test + '_predictions'][key] = torch.cat(model_outputs, dim=0).numpy()
        
        results['total_losses'][key] = np.asarray(total_losses)

    results['test_performance'] = plot_decoder_predictions(results['decoder_predictions'], labels_dict, results['test_performance'], display_plots, save_plots=save_plots, save_path_prefix='outputs/' if save_plots else None, model_name=model_name)
    #results['decoder_predictions'] = torch.cat(model_outputs, dim=0).numpy()
    #labels_dict = testset['videos']
    #results['test_performance'] = {}
    #display_plots = True

    print("Testing completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return results


def compute_saliency(model, model_input, ground_truth, device):
    """
    Compute saliency map for the input.
    
    Args:
    - model: PyTorch model, the neural network that outputs an image of shape (3, 112, 112)
    - model_input: Input tensor of shape (1, N), representing the input data
    - ground_truth: Ground truth tensor of shape (1, 3, 112, 112), representing the reference image
    - device: Device to perform the computations on, 'cuda' or 'cpu'
    
    Returns:
    - slc: Saliency vector of shape (4600), representing the gradients of SSIM with respect to the input
    """

    model = model.to(device)
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move inputs to the appropriate device
    model_input = model_input.to(device).requires_grad_(True)  # Ensure input requires gradients
    ground_truth = ground_truth.to(device)

    # Forward pass to get the model output
    output = model(model_input)  # Should output an image tensor of shape (1, 3, 112, 112)
    
    # Compute SSIM between model output and ground truth
    ssim_value = ssim(normalize(output), normalize(ground_truth), data_range=1.0)  # Assuming inputs are normalized [0, 1]
    
    # Backward pass to compute gradients
    ssim_value.backward()  # Compute gradients of SSIM with respect to the input
    slc = np.abs(model_input.grad.data.cpu().numpy().flatten())

    return slc





# training on many subjects instead of average


from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import random
import gc

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc

class SingleSubjectDataset(Dataset):
    """Dataset that loads data for a single subject."""
    
    def __init__(self, subject_folder, videos_folder, train_fraction=0.8, test_video=None):
        """
        Initialize dataset for a single subject.
        
        Args:
            subject_folder (str): Path to subject folder
            videos_folder (str): Path to folder with processed video data
            train_fraction (float): Fraction of timepoints to use from each video (0-1)
            test_video (str, optional): Name of video file to exclude from training
        """
        self.subject_folder = subject_folder
        self.videos_folder = videos_folder
        self.train_fraction = train_fraction
        self.subject_id = os.path.basename(subject_folder)
        
        # Build a mapping of videos and timepoints
        self.video_paths = []  # List of video_names
        self.timepoints = []   # List of (video_idx, timepoint) tuples
        
        # Process all videos for this subject
        print(f"Scanning subject {self.subject_id}...")
        total_videos = 0
        total_timepoints = 0
        
        # Get all video files for this subject
        try:
            video_files = [f for f in os.listdir(subject_folder) if f.endswith('.npy')]
            video_files.sort()  # Sort for reproducibility
            
            # Remove test video if specified
            if test_video is not None and test_video in video_files:
                video_files.remove(test_video)
                
            # Process each video
            for video_name in video_files:
                # Check if both fMRI and label files exist
                fmri_path = os.path.join(subject_folder, video_name)
                label_path = os.path.join(videos_folder, video_name)
                
                if not os.path.exists(fmri_path):
                    print(f"WARNING: fMRI file not found: {fmri_path}")
                    continue
                    
                if not os.path.exists(label_path):
                    print(f"WARNING: Label file not found: {label_path}")
                    continue
                
                try:
                    # Load and prepare the fMRI data as in your working code
                    fmri_data = np.load(fmri_path, mmap_mode='r')
                    label_data = np.load(label_path, mmap_mode='r')
                    
                    # Process fMRI data with load_durations as you specified
                    subject_id_short = self.subject_id[4:] if self.subject_id.startswith('sub-') else self.subject_id
                    video_name_short = video_name[:-4] if video_name.endswith('.npy') else video_name
                    
                    onset, duration = load_durations(video=video_name_short, subject=subject_id_short)
                    
                    # Apply necessary slicing and transposing
                    fmri_shape_before = fmri_data.shape
                    fmri_data = fmri_data[:,onset:onset+duration]
                    fmri_data = np.transpose(fmri_data,(1,0))
                    
                    if total_videos < 2:  # Print diagnostics for a couple of videos
                        print(f"  Applied onset/duration processing to {video_name}:")
                        print(f"    Shape before: {fmri_shape_before}, after: {fmri_data.shape}")
                        print(f"    Used onset={onset}, duration={duration}")
                    
                    # Get the TRs from the first dimensions
                    fmri_trs = fmri_data.shape[0]
                    label_trs = label_data.shape[0]
                    
                    # Verify they match
                    if fmri_trs != label_trs:
                        print(f"WARNING: TR mismatch for {video_name}: fMRI TRs={fmri_trs}, label TRs={label_trs}")
                        continue
                        
                    video_length = fmri_trs
                    timepoints_to_use = int(video_length * train_fraction)
                    
                    # Add video to the list
                    video_idx = len(self.video_paths)
                    self.video_paths.append(video_name)
                    
                    # Add timepoints for this video
                    for t in range(timepoints_to_use):
                        self.timepoints.append((video_idx, t))
                        
                    total_videos += 1
                    total_timepoints += timepoints_to_use
                    
                except Exception as e:
                    print(f"ERROR processing {video_name}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"ERROR processing subject {self.subject_id}: {str(e)}")
            
        print(f"Subject {self.subject_id}: {total_videos} videos, {total_timepoints} timepoints")
        print(f"Using first {train_fraction*100:.0f}% of timepoints from each video")
        
        # Cache for the currently loaded video
        self.current_video_idx = None
        self.current_fmri = None
        self.current_label = None
        
    def __len__(self):
        return len(self.timepoints)
    
    def __getitem__(self, idx):
        # Get the video and timepoint
        video_idx, timepoint = self.timepoints[idx]
        video_name = self.video_paths[video_idx]
        
        # Check if we need to load a new video
        if self.current_video_idx != video_idx:
            # Load fMRI data
            fmri_path = os.path.join(self.subject_folder, video_name)
            fmri_data = np.load(fmri_path)
            
            # Load label data
            label_path = os.path.join(self.videos_folder, video_name)
            label_data = np.load(label_path)
            
            # Process fMRI data with load_durations as you specified
            subject_id_short = self.subject_id[4:] if self.subject_id.startswith('sub-') else self.subject_id
            video_name_short = video_name[:-4] if video_name.endswith('.npy') else video_name
            
            onset, duration = load_durations(video=video_name_short, subject=subject_id_short)
            fmri_data = fmri_data[:,onset:onset+duration]
            fmri_data = np.transpose(fmri_data,(1,0))
            
            # Store data
            self.current_fmri = fmri_data
            self.current_label = label_data
            self.current_video_idx = video_idx
            
            # Verify shapes on first access
            if idx == 0:
                print(f"Loaded video {video_name}:")
                print(f"  fMRI shape: {self.current_fmri.shape}")
                print(f"  Label shape: {self.current_label.shape}")
        
        # Get the timepoint data
        fmri_data = self.current_fmri[timepoint]
        label_data = self.current_label[timepoint]
        
        # Convert to tensors
        fmri_tensor = torch.from_numpy(fmri_data)
        label_tensor = torch.from_numpy(label_data)
        
        return fmri_tensor, label_tensor
    
    def clear_cache(self):
        """Clear the cached data to free memory"""
        self.current_video_idx = None
        self.current_fmri = None
        self.current_label = None
        gc.collect()


def train_model2(subject_folders, videos_folder, model, num_epochs, lr, criterion, optimizer, batch_size, 
               device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, 
               model_to_train=None, display_plots=True, save_plots=False, model_name="",
               train_fraction=0.8, test_video=None, checkpoint_every=5, subjects_per_batch=1):
    """
    Train the model by processing subjects sequentially to manage memory usage.

    Arguments:
        subject_folders (list): List of paths to subject fMRI folders
        videos_folder (str): Path to the processed videos folder
        model (nn.Module): The neural network model to be trained
        num_epochs (int): Number of epochs to train the model
        lr (float): Learning rate for the optimizer
        criterion (nn.Module): Loss function for training
        optimizer (torch.optim.Optimizer): Optimizer for model training
        batch_size (int): Batch size for training
        device (torch.device): Device to train the model on (CPU or GPU)
        save_model_as (str): Path to save the trained model
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'
        train_fraction (float): Fraction of timepoints in each video to use for training (0-1), default is 0.8 (80%)
        test_video (str, optional): Name of a specific video to reserve for testing only
        subjects_per_batch (int): Number of subjects to process together before releasing memory

    Returns:
        model (nn.Module): Trained model
        history (dict): Dictionary containing training loss history
    """
    import os
    import gc
    import torch
    import numpy as np
    import time
    
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_train not in model_type:
        print(f'model_to_train: {model_to_train} not recognized. Must be one of {model_type}')
        return None, None

    print(f'### Training {model_to_train} using train_model2 on {len(subject_folders)} subjects sequentially ###')
    print(f'### Using first {train_fraction*100:.0f}% of timepoints from each video ###')
    
    # Load pretrained decoder if provided
    if pretrained_decoder:
        decoder = None  # Will be initialized once we know data shape
        print(f'Will use pretrained decoder {pretrained_decoder}')

    # Move model and criterion to device
    model = model.to(device)
    criterion = criterion.to(device)

    # Initialize history dictionary
    history = {
        'total_loss': [],
        'other_metrics': [],
        'metrics_names': [],
        'subject_epochs': {}  # Track which subjects were used in each epoch
    }
    
    # Main training loop
    for epoch in range(start_epoch, num_epochs+1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_metrics = None
        epoch_subjects = []
        
        # Process subjects in groups to manage memory
        for i in range(0, len(subject_folders), subjects_per_batch):
            batch_subject_folders = subject_folders[i:i+subjects_per_batch]
            print(f"\nProcessing subjects {i+1}-{i+len(batch_subject_folders)} of {len(subject_folders)}")
            
            # Train on each subject in this batch
            for subject_folder in batch_subject_folders:
                subject_id = os.path.basename(subject_folder)
                print(f"\nTraining on subject {subject_id}")
                epoch_subjects.append(subject_id)
                
                # Create dataset for this subject
                dataset = SingleSubjectDataset(
                    subject_folder=subject_folder,
                    videos_folder=videos_folder,
                    train_fraction=train_fraction,
                    test_video=test_video
                )
                
                # Skip if no data for this subject
                if len(dataset) == 0:
                    print(f"No valid data for subject {subject_id}, skipping")
                    continue
                
                # Create dataloader
                train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=False,
                    num_workers=2,  # Fewer workers to reduce memory usage
                )
                
                # Initialize pretrained decoder if needed
                if pretrained_decoder and decoder is None:
                    # Get sample data to determine shape
                    sample_inputs, sample_labels = next(iter(train_loader))
                    if model_to_train == 'decoder':
                        input_shape = sample_labels.shape[1]
                    else:
                        input_shape = sample_inputs.shape[1]
                    
                    # Initialize decoder
                    from models_new_2 import Decoder  # Import where needed
                    decoder = Decoder(input_shape)
                    state_dict = torch.load(pretrained_decoder)
                    decoder.load_state_dict(state_dict)
                    decoder = decoder.to(device)
                    for param in decoder.parameters():
                        param.requires_grad = False
                    decoder.eval()
                    print(f'Initialized decoder with input shape {input_shape}')
                
                # Train on this subject
                for j, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass based on model type
                    if model_to_train == 'encoder_decoder':
                        model_outputs, decoder_outputs = model(inputs.float())
                    else:
                        model_outputs = model(inputs.float())
                        if pretrained_decoder:
                            decoder_outputs = decoder(model_outputs.float())
                        else:
                            decoder_outputs = None
                    
                    # Apply criterion
                    if model_to_train == 'decoder':
                        # Take the middle frame as label
                        if labels.dim() > 2 and labels.shape[-1] > 15:
                            *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels[..., 15])
                        else:
                            *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
                    elif decoder_outputs is None:
                        *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
                    else:
                        # Middle frame
                        if inputs.dim() > 2 and inputs.shape[-1] > 15:
                            *loss_metrics, total_loss, metrics_names = criterion(
                                model_outputs, labels, decoder_outputs, inputs[..., 15]
                            )
                        else:
                            *loss_metrics, total_loss, metrics_names = criterion(
                                model_outputs, labels, decoder_outputs, inputs
                            )
                    
                    # Backward pass and optimize
                    total_loss.backward()
                    optimizer.step()
                    running_loss += total_loss.item()
                    batch_count += 1
                    
                    # Store metrics
                    if epoch_metrics is None:
                        epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
                    else:
                        epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                      for j, metric in enumerate(loss_metrics)]
                    
                    # Print progress
                    if j % 20 == 0:
                        print(f"Subject {subject_id}, Batch {j}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
                
                # Clear this subject's data from memory
                dataset.clear_cache()
                del train_loader
                del dataset
                gc.collect()
                print(f"Cleared memory after processing subject {subject_id}")
            
            # Force garbage collection after processing a batch of subjects
            gc.collect()
            print(f"Completed subjects {i+1}-{i+len(batch_subject_folders)}")
        
        # Calculate epoch statistics
        avg_loss = running_loss / max(batch_count, 1)
        avg_metrics = [metric / max(batch_count, 1) for metric in epoch_metrics] if epoch_metrics else []
        
        # Print epoch summary
        print(f"\nEpoch {epoch} complete - Avg Loss: {avg_loss:.4f} - Trained on subjects: {', '.join(epoch_subjects)}")
        
        # Save checkpoint
        if epoch % checkpoint_every == 0 and display_plots:
            save_checkpoint(model, optimizer, epoch, avg_loss)
        
        # Store history
        history['total_loss'].append(avg_loss)
        history['other_metrics'].append(avg_metrics)
        history['metrics_names'] = metrics_names
        history['subject_epochs'][epoch] = epoch_subjects
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"Memory cleaned after epoch {epoch}")
    
    # Save final model
    torch.save(model.state_dict(), save_model_as)
    
    # Convert history to numpy arrays
    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics']) if len(history['other_metrics']) > 0 else []
    
    # Plot if requested
    if display_plots:
        plot_train_losses(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history





def train_model3(base_dir, subjects_list, videos_path, model, num_epochs, lr, criterion, 
                              optimizer, batch_size, device, save_model_as, pretrained_decoder=None, 
                              start_epoch=1, start_loss=None, model_to_train=None, 
                              display_plots=True, save_plots=True, model_name=""):
    """
    Train the model using data from multiple subjects.
    
    Arguments:
        base_dir (str): Base directory containing subject folders
        subjects_list (list): List of subject folder names (e.g., ['sub-S01', 'sub-S02', ...])
        videos_path (str): Path to the videos.npy file (labels)
        model (nn.Module): The neural network model to be trained
        ... (other parameters same as original train_model function)
    
    Returns:
        model (nn.Module): Trained model
        history (dict): Dictionary containing training loss history
    """
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_train not in model_type:
        print(f'model_to_train: {model_to_train} not recognized. Must be one of {model_type}')
        return None, None
    
    # Load labels (videos) once since they're the same for all subjects
    print(f'Loading labels from {videos_path}')
    labels = np.load(videos_path)
    
    # Initialize pretrained decoder if provided
    if pretrained_decoder:
        decoder = Decoder(labels.shape[1])
        state_dict = torch.load(pretrained_decoder)
        decoder.load_state_dict(state_dict)
        decoder = decoder.to(device)
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        print(f'Also using pretrained decoder {pretrained_decoder}')

    print(f"Start training from epoch {start_epoch} with initial loss {start_loss}")
    
    # Move model to device
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Initialize history dictionary
    history = {
        'total_loss': [],
        'other_metrics': [],
        'metrics_names': []
    }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs+1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_metrics = None
        
        # Each epoch, we process all subjects
        for subject in subjects_list:
            subject_path = os.path.join(base_dir, subject, 'train.npy')
            print(f'Processing {subject} from {subject_path}')
            
            try:
                # Load data for this subject
                inputs = np.load(subject_path)

                
                # Create dataset and dataloader for this subject
                inputs_tensor = torch.from_numpy(inputs)
                labels_tensor = torch.from_numpy(labels)
                
                # Print shapes for debugging
                print(f'Subject {subject} - Input shape: {inputs_tensor.shape}, Label shape: {labels_tensor.shape}')
                
                subject_dataset = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
                subject_loader = torch.utils.data.DataLoader(
                    subject_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=False,
                    num_workers=4,
                )
                
                # Train on this subject's data
                for i, (inputs_batch, labels_batch) in enumerate(subject_loader):
                    inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
                    optimizer.zero_grad()

                    # Determine the outputs based on the model configuration
                    if model_to_train == 'encoder_decoder':
                        model_outputs, decoder_outputs = model(inputs_batch.float())
                    else:
                        model_outputs = model(inputs_batch.float())
                        if pretrained_decoder:
                            # If there's a pretrained decoder, use it with encoder outputs
                            decoder_outputs = decoder(model_outputs.float())
                        else:
                            # If no pretrained decoder, proceed with encoder outputs as main outputs
                            decoder_outputs = None
                
                    # Apply the appropriate criterion based on the presence of decoder outputs
                    if model_to_train == 'decoder':
                        *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels_batch[..., 15])
                    elif decoder_outputs is None:
                        *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels_batch)
                    else:
                        *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels_batch, decoder_outputs, inputs_batch[..., 15])

                    total_loss.backward()
                    optimizer.step()
                    running_loss += total_loss.item()
                    batch_count += 1

                    # Store metrics for averaging later
                    if epoch_metrics is None:
                        epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
                    else:
                        epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                        for j, metric in enumerate(loss_metrics)]
            
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
                continue
        
        # Calculate average loss and metrics for the epoch across all subjects
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []
            
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")

            if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
#                print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
                save_checkpoint(model, optimizer, epoch+1, avg_loss)

            history['total_loss'].append(avg_loss)
            history['other_metrics'].append(avg_metrics)
        else:
            print(f"Warning: No valid batches processed in epoch {epoch}")
    
    # Save history and model
    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names

    if display_plots:
        try:
            plot_train_losses(history, start_epoch, save_plots=save_plots, 
                            save_path=f'outputs/training_{model_name}.png' if save_plots else None)
        except:
            print(f"Error saving to original location: {e}")
        
            # Create an outputs directory in /media/ if it doesn't exist
            media_outputs_path = '/media/RCPNAS/MIP/Michael/students_work/rodrigo/outputs2'
            try:
                os.makedirs(media_outputs_path, exist_ok=True)
                print(f"Created alternative save location: {media_outputs_path}")
                
                # Try to save in the new location
                plot_train_losses(history, start_epoch, save_plots=save_plots, 
                                save_path=f'{media_outputs_path}/training_{model_name}.png' if save_plots else None)
                print(f"Successfully saved plot to alternative location")
            except Exception as e2:
                print(f"Failed to save to alternative location: {e2}")
                # If you still want to display the plot even if saving fails
                plot_train_losses(history, start_epoch, save_plots=False)


    torch.save(model.state_dict(), save_model_as)
    
#    if display_plots:
#        plot_train_losses(history, start_epoch, save_plots=save_plots, 
#                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history




def train_model3_quicktest(base_dir, subjects_list, videos_path, model, num_epochs, lr, criterion, 
                              optimizer, batch_size, device, save_model_as, pretrained_decoder=None, 
                              start_epoch=1, start_loss=None, model_to_train=None, 
                              display_plots=True, save_plots=True, model_name=""):
    """
    Train the model using data from multiple subjects.
    
    Arguments:
        base_dir (str): Base directory containing subject folders
        subjects_list (list): List of subject folder names (e.g., ['sub-S01', 'sub-S02', ...])
        videos_path (str): Path to the videos.npy file (labels)
        model (nn.Module): The neural network model to be trained
        ... (other parameters same as original train_model function)
    
    Returns:
        model (nn.Module): Trained model
        history (dict): Dictionary containing training loss history
    """
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_train not in model_type:
        print(f'model_to_train: {model_to_train} not recognized. Must be one of {model_type}')
        return None, None
    
    # Load labels (videos) once since they're the same for all subjects
    print(f'Loading labels from {videos_path}')
    #labels = np.load(videos_path)
    
    full_labels = np.load(videos_path)
    print("full_labels shape:", full_labels.shape)
    first_10_l = full_labels[:10, ...]  # First 10 samples (with all their features)
    last_10_l = full_labels[-10:, ...]  # Last 10 samples (with all their features)
    labels = np.concatenate([first_10_l, last_10_l], axis=0)  # Combine them
    
    # Initialize pretrained decoder if provided
    if pretrained_decoder:
        decoder = Decoder(full_labels.shape[1])
        state_dict = torch.load(pretrained_decoder)
        decoder.load_state_dict(state_dict)
        decoder = decoder.to(device)
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        print(f'Also using pretrained decoder {pretrained_decoder}')

    print(f"Start training from epoch {start_epoch} with initial loss {start_loss}")
    
    # Move model to device
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Initialize history dictionary
    history = {
        'total_loss': [],
        'other_metrics': [],
        'metrics_names': []
    }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs+1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_metrics = None
        
        # Each epoch, we process all subjects
        for subject in subjects_list:
            
            subject_path = os.path.join(base_dir, subject, 'train.npy')
            print(f'Processing {subject} from {subject_path}')
            
            try:
                # Load data for this subject
                # inputs = np.load(subject_path)
                full_inputs = np.load(subject_path)
                print("full_inputs shape:", full_inputs.shape, "ta chill se a primeira dimensao nao for os 4609 e for igual a primeira dos videos")
                first_10 = full_inputs[:10, ...]  # First 10 samples (with all their features)
                last_10 = full_inputs[-10:, ...]  # Last 10 samples (with all their features)
                inputs = np.concatenate([first_10, last_10], axis=0)  # Combine them

                
                # Create dataset and dataloader for this subject
                inputs_tensor = torch.from_numpy(inputs)
                labels_tensor = torch.from_numpy(labels)
                
                # Print shapes for debugging
                print(f'Subject {subject} - Input shape: {inputs_tensor.shape}, Label shape: {labels_tensor.shape}')
                
                subject_dataset = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
                subject_loader = torch.utils.data.DataLoader(
                    subject_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=False,
                    num_workers=4,
                )
                
                # Train on this subject's data

                batch_count_subject = 0

                for i, (inputs_batch, labels_batch) in enumerate(subject_loader):
                    inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
                    optimizer.zero_grad()

                    if batch_count_subject == 1:
                        print(f"inputs_batch shape: {inputs_batch.shape}, labels_batch shape: {labels_batch.shape}")
                    
                    #plottar cerebro do input que eu escolhi
                    #plottar cerebro desse input no average antigo ou seja do encoder_dataset_6661 ou entao nao vai dar por causa do validation set
                    #ok sq tenho de perceber mm onde esta cada video no encoder_dataset mas por enquanto vou so fazer com a primeira imagem dos dois se conseguir, se bem que isso nao me deve dizer grande coisa sobre como a data foi juntada, so sobre se pelo menos a primeira frame tem atividade semelhante


                    # Determine the outputs based on the model configuration
                    if model_to_train == 'encoder_decoder':
                        model_outputs, decoder_outputs = model(inputs_batch.float())
                    else:
                        model_outputs = model(inputs_batch.float())
                        if pretrained_decoder:
                            # If there's a pretrained decoder, use it with encoder outputs
                            decoder_outputs = decoder(model_outputs.float())
                        else:
                            # If no pretrained decoder, proceed with encoder outputs as main outputs
                            decoder_outputs = None
                
                    # Apply the appropriate criterion based on the presence of decoder outputs
                    if model_to_train == 'decoder':
                        *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels_batch[..., 15])
                    elif decoder_outputs is None:
                        *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels_batch)
                    else:
                        *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels_batch, decoder_outputs, inputs_batch[..., 15])

                    total_loss.backward()
                    optimizer.step()
                    running_loss += total_loss.item()
                    batch_count += 1
                    batch_count_subject += 1

                    # Store metrics for averaging later
                    if epoch_metrics is None:
                        epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
                    else:
                        epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                        for j, metric in enumerate(loss_metrics)]
            
            except Exception as e:
                print(f"Error processing subject {subject}: {e}")
                continue
        
        # Calculate average loss and metrics for the epoch across all subjects
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []
            
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")

            if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
#                print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
                save_checkpoint(model, optimizer, epoch+1, avg_loss)

            history['total_loss'].append(avg_loss)
            history['other_metrics'].append(avg_metrics)
        else:
            print(f"Warning: No valid batches processed in epoch {epoch}")
    
    # Save history and model
    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names

    if display_plots:
        try:
            plot_train_losses(history, start_epoch, save_plots=save_plots, 
                            save_path=f'outputs/training_{model_name}.png' if save_plots else None)
        except:
            print(f"Error saving to original location: {e}")
        
            # Create an outputs directory in /media/ if it doesn't exist
            media_outputs_path = '/media/RCPNAS/MIP/Michael/students_work/rodrigo/outputs2'
            try:
                os.makedirs(media_outputs_path, exist_ok=True)
                print(f"Created alternative save location: {media_outputs_path}")
                
                # Try to save in the new location
                plot_train_losses(history, start_epoch, save_plots=save_plots, 
                                save_path=f'{media_outputs_path}/training_{model_name}.png' if save_plots else None)
                print(f"Successfully saved plot to alternative location")
            except Exception as e2:
                print(f"Failed to save to alternative location: {e2}")
                # If you still want to display the plot even if saving fails
                plot_train_losses(history, start_epoch, save_plots=False)


    torch.save(model.state_dict(), save_model_as)
    
#    if display_plots:
#        plot_train_losses(history, start_epoch, save_plots=save_plots, 
#                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history





class Decoder_Temporal2(nn.Module):
    def __init__(self, mask_size, temporal_window=3):
        """
        Initialize the Decoder_Temporal2 architecture with temporal processing.
        
        Arguments:
            mask_size (int): The size of the input layer, corresponding to the size of the fMRI mask in use.
            temporal_window (int): Number of consecutive TRs to process together.
        """
        super(Decoder_Temporal2, self).__init__()
        
        self.temporal_window = temporal_window
        
        # Temporal processing layer - processes multiple TRs
        if temporal_window > 1:
            self.temporal_conv = nn.Conv1d(mask_size, mask_size, kernel_size=temporal_window, 
                                          padding=temporal_window//2)
            self.temporal_bn = nn.BatchNorm1d(mask_size)
        
        # Main decoder architecture (same as before)
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
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, x):
        """
        Define the forward pass of the Decoder_Temporal2.
        
        Arguments:
            x (Tensor): The input data to the Decoder_Temporal2.
                        If temporal_window=1: shape (batch_size, mask_size)
                        If temporal_window>1: shape (batch_size, temporal_window, mask_size)
        
        Returns:
            Tensor: The decoded output (shape of (batch_size, 3, 112, 112)).
        """
        # Handle temporal input if provided
        if self.temporal_window > 1 and x.dim() == 3:
            batch_size, temp_window, mask_size = x.size()
            
            # Transpose to (batch_size, mask_size, temporal_window) for 1D conv
            x = x.transpose(1, 2)
            
            # Apply temporal convolution to capture patterns across TRs
            x = self.temporal_conv(x)
            x = F.relu(x)
            x = self.temporal_bn(x)
            
            # Extract features from the middle position (represents the current TR)
            mid_idx = temp_window//2
            x = x[:, :, mid_idx]
        
        # Fully connected layer
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(-1, 48, 14, 14) # Reshape to (batch_size, channels, H, W)
        
        # First conv layer + ReLU + Upsample + BatchNorm
        x = self.conv1(x)
        x = F.relu(x)
        x = self.up1(x)
        x = self.bn1(x)

        # Second conv layer + ReLU + Upsample + BatchNorm
        x = self.conv2(x)
        x = F.relu(x)
        x = self.up2(x)
        x = self.bn2(x)

        # Third conv layer + ReLU + Upsample + BatchNorm
        x = self.conv3(x)
        x = F.relu(x)
        x = self.up3(x)
        x = self.bn3(x)
        
        # Fourth conv layer + sigmoid + BatchNorm
        x = self.conv4(x)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        return x



'''
def train_model_temporal(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name="", temporal_window=3):
    """
    Train the model using the specified parameters and dataset with temporal window.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', 
                           then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', 
                           then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        temporal_window (int, optional): Number of consecutive TRs to use as input for temporal processing. Default is 3.

    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training loss history.
    """
    tic = time.time()

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

    model_type = ['encoder', 'decoder', 'encoder_decoder']
    if model_to_train not in model_type:
        print(f'model_to_train: {model_to_train} not recognized. Must be one of {model_type}')
        return None, None

    print(f'### Training {model_to_train} on input of shape {input.shape} with temporal window {temporal_window} ###')
    if pretrained_decoder:
        decoder = Decoder_Temporal2(label.shape[1])
        state_dict = torch.load(pretrained_decoder)
        decoder.load_state_dict(state_dict)
        decoder = decoder.to(device)
        for param in decoder.parameters():
            param.requires_grad = False
        decoder.eval()
        print(f'Also using pretrained decoder {pretrained_decoder}')

    print(f"Start training from epoch {start_epoch} with initial loss {start_loss}")
    
    # Convert to torch tensors
    input = torch.from_numpy(input)
    label = torch.from_numpy(label)
    
    # Prepare data with temporal windows if applicable
    if temporal_window > 1 and model_to_train == 'decoder':
        # For decoder, prepare temporal windows of fMRI data
        temporal_inputs = []
        temporal_labels = []
        
        for i in range(len(input) - temporal_window + 1):
            # Create a window of 'temporal_window' consecutive TRs
            window = input[i:i+temporal_window]
            
            # Use the frame from the middle TR as the target
            middle_idx = i + temporal_window // 2
            target = label[middle_idx]
            
            temporal_inputs.append(window)
            temporal_labels.append(target)
        
        # Convert to tensors
        temporal_inputs = torch.stack(temporal_inputs)
        temporal_labels = torch.stack(temporal_labels)
        
        # Create dataset with temporal windows
        train_set = torch.utils.data.TensorDataset(temporal_inputs, temporal_labels)
        print(f"Created temporal dataset with {len(temporal_inputs)} samples")
    else:
        # For encoder or encoder_decoder, or if temporal_window=1, use original data
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
        batch_count = 0
        epoch_metrics = None

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
        
            # Apply the appropriate criterion
            if model_to_train == 'decoder':
                # For decoder, use labels directly (middle frame of window)
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                # For encoder_decoder, need to adjust for temporal_window if used
                if temporal_window > 1:
                    # Use middle frame of input window
                    mid_frame_idx = temporal_window // 2
                    input_frame = inputs[:, mid_frame_idx, ..., 15] if inputs.dim() > 4 else inputs[..., 15]
                else:
                    input_frame = inputs[..., 15]  # Middle frame from original inputs
                
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, input_frame)

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                               for j, metric in enumerate(loss_metrics)]

        # Calculate average loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        if epoch % 5 == 0 and display_plots:
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)

        history['total_loss'].append(avg_loss)
        history['other_metrics'].append(avg_metrics)

    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    if display_plots:
        plot_train_losses(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history
'''


def train_model_temporal(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name=""):
    """
    Train the model using the specified parameters and dataset.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.

    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training loss history.
    """
    tic = time.time()

    print("input at beginning of function shape =", input.shape)
    print("label at beginning of function shape =", label.shape)
#    time.sleep(10)

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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

    print("input after .from_numpy shape =", input.shape)
    print("label after .from_numpy shape =", label.shape)
    

    train_set = torch.utils.data.TensorDataset(input, label)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=4,
    )

#    print("\n\ntrain_loader shape =", train_loader.shape)
#    print("train_loader.dataset[0] shape =", train_loader.dataset[0].shape)
#    print("train_loader.dataset[0][1] shape =", train_loader.dataset[0][1].shape, "\n\n")
    #time.sleep(10)

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
        batch_count = 0
        epoch_metrics = None

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                print("inputs shape to generate model_output =", inputs.shape)
                #inputs shape to generate model_output = torch.Size([16, 3, 4609]
                model_outputs = model(inputs.float())       #???
                print("model_outputs shape just after generating =", model_outputs.shape)
                #model_outputs shape just after generating = torch.Size([16, 3, 3, 112, 112])
                # time.sleep(10)
                if pretrained_decoder:
                    # If there's a pretrained decoder, use it with encoder outputs
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    # If no pretrained decoder, proceed with encoder outputs as main outputs
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':

                print("model_outputs shape =", model_outputs.shape)
                print("labels shape =", labels.shape, "\n")
                #time.sleep(10)
                # model_outputs shape [16, 3, 3, 112, 112]
                # label shape [4319, 3, 3, 112, 112])
                # acho que me enganei aqui pq o output agr ta a dar labels shape = torch.Size([16, 3, 3, 112, 112])
                

                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)          #--> take the middle frame as label
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, torch.mean(labels, dim=4)) #--> take the average frame as label
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])          #--> middle frame
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, torch.mean(inputs, dim=4)) #--> average frame

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                for j, metric in enumerate(loss_metrics)]

            # Store metrics for averaging later
#            if epoch_metrics is None:
#                epoch_metrics = [metric.item() for metric in loss_metrics]
#            else:
#                epoch_metrics = [epoch_metrics[j] + metric.item() for j, metric in enumerate(loss_metrics)]

        # Calculate average loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
#            print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss / (5*i):.4f}")
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)
#            running_loss = 0.0
#            save_checkpoint(model, optimizer, epoch+1, total_loss)

#        history['total_loss'].append(total_loss.item())
#        history['other_metrics'].append(loss_metrics)  # Store other metrics for visualization

        history['total_loss'].append(avg_loss)  # Store average loss instead of just the last batch's loss
        history['other_metrics'].append(avg_metrics)  # Store average metrics for visualization

    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    if display_plots:
#        plot_train_losses(history, start_epoch)
        plot_train_losses(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history






'''def l1_regularization_loss(model, lambda_l1):
    """
    Calculate L1 regularization loss for all parameters in the model.
    
    Args:
        model: PyTorch model
        lambda_l1: L1 regularization strength
    
    Returns:
        L1 regularization loss term
    """
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss'''



def l1_regularization_loss(model, lambda_l1):
    """
    Calculate L1 regularization loss for all parameters in the model.
    
    Args:
        model: PyTorch model
        lambda_l1: L1 regularization strength
    
    Returns:
        L1 regularization loss term
    """
    l1_loss = 0
    total_params = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
      #    total_params += param.numel()  # Count number of parameters
    
    # Normalize by number of parameters
      #l1_loss = l1_loss / total_params
    return lambda_l1 * l1_loss



'''
def train_model_temporal_with_val_and_reg(input, label, val_input, val_label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name="", lambda_l1=None):
    """
    Train the model using the specified parameters and dataset.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        val_input (numpy array): Input validation data to the model. If training a decoder, shape of (TR, mask_size).
        val_label (numpy array): Target validation labels for the input data. If training a decoder, shape of (TR, 3, 112, 112, 32).
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        lambda_l1 (float, optional): Regularization strength for L1 regularization. Default is None.
        
    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training loss history.
    """
    tic = time.time()

    print("input at beginning of function shape =", input.shape)
    print("label at beginning of function shape =", label.shape)
#    time.sleep(10)

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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

    print("input after .from_numpy shape =", input.shape)
    print("label after .from_numpy shape =", label.shape)
    

    train_set = torch.utils.data.TensorDataset(input, label)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=4,
    )

#    print("\n\ntrain_loader shape =", train_loader.shape)
#    print("train_loader.dataset[0] shape =", train_loader.dataset[0].shape)
#    print("train_loader.dataset[0][1] shape =", train_loader.dataset[0][1].shape, "\n\n")
    #time.sleep(10)

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
        batch_count = 0
        epoch_metrics = None

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                print("inputs shape to generate model_output =", inputs.shape)
                #inputs shape to generate model_output = torch.Size([16, 3, 4609]
                model_outputs = model(inputs.float())       #???
                print("model_outputs shape just after generating =", model_outputs.shape)
                #model_outputs shape just after generating = torch.Size([16, 3, 3, 112, 112])
                # time.sleep(10)
                if pretrained_decoder:
                    # If there's a pretrained decoder, use it with encoder outputs
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    # If no pretrained decoder, proceed with encoder outputs as main outputs
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':

                print("model_outputs shape =", model_outputs.shape)
                print("labels shape =", labels.shape, "\n")
                #time.sleep(10)
                # model_outputs shape [16, 3, 3, 112, 112]
                # label shape [4319, 3, 3, 112, 112])
                # acho que me enganei aqui pq o output agr ta a dar labels shape = torch.Size([16, 3, 3, 112, 112])
                

                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)          #--> take the middle frame as label
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, torch.mean(labels, dim=4)) #--> take the average frame as label
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])          #--> middle frame
                #*loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, torch.mean(inputs, dim=4)) #--> average frame

            #apply regularization
            if lambda_l1 is not None:
                l1_loss = l1_regularization_loss(model, lambda_l1)
                total_loss = total_loss + l1_loss

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                for j, metric in enumerate(loss_metrics)]

            # Store metrics for averaging later
#            if epoch_metrics is None:
#                epoch_metrics = [metric.item() for metric in loss_metrics]
#            else:
#                epoch_metrics = [epoch_metrics[j] + metric.item() for j, metric in enumerate(loss_metrics)]

        # Calculate average loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
#            print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss / (5*i):.4f}")
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)
#            running_loss = 0.0
#            save_checkpoint(model, optimizer, epoch+1, total_loss)

#        history['total_loss'].append(total_loss.item())
#        history['other_metrics'].append(loss_metrics)  # Store other metrics for visualization

        history['total_loss'].append(avg_loss)  # Store average loss instead of just the last batch's loss
        history['other_metrics'].append(avg_metrics)  # Store average metrics for visualization

    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    if display_plots:
#        plot_train_losses(history, start_epoch)
        plot_train_losses(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history
'''




def train_model_temporal_with_val_and_reg_og(input, label, val_input_dict, val_label_dict, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name="", lambda_l1=None):
    """
    Train the model using the specified parameters and dataset.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        val_input_dict (dict): Dictionary of validation input data. Keys are movie names, same structure as inputs_dict in test_model.
        val_label_dict (dict): Dictionary of validation labels. Keys are movie names, same structure as labels_dict in test_model.
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        lambda_l1 (float, optional): Regularization strength for L1 regularization. Default is None.
        
    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training loss history.
    """
    tic = time.time()

    print("input at beginning of function shape =", input.shape)
    print("label at beginning of function shape =", label.shape)
    print("val_input_dict keys =", list(val_input_dict.keys()))
    print("val_label_dict keys =", list(val_label_dict.keys()))

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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
    
    # Prepare training data
    input = torch.from_numpy(input)
    label = torch.from_numpy(label)

    print("input after .from_numpy shape =", input.shape)
    print("label after .from_numpy shape =", label.shape)
    
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
        'validation_loss': [],  # Add validation loss tracking
        'other_metrics': [],
        'metrics_names': []
    }
    
    for epoch in range(start_epoch, num_epochs+1):
        # Training phase
        model.train()                                   
        running_loss = 0.0
        batch_count = 0
        epoch_metrics = None

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                #print("inputs shape to generate model_output =", inputs.shape)
                model_outputs = model(inputs.float())
                #print("model_outputs shape just after generating =", model_outputs.shape)
                if pretrained_decoder:
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':
                #print("model_outputs shape =", model_outputs.shape)
                #print("labels shape =", labels.shape, "\n")
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])

            #apply regularization
            if lambda_l1 is not None:
                l1_loss = l1_regularization_loss(model, lambda_l1)
                total_loss = total_loss + l1_loss

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                for j, metric in enumerate(loss_metrics)]

        # Calculate average training loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        # Validation phase (run every 5 epochs)
        avg_val_loss = None
        if epoch % 5 == 0:
            model.eval()
            val_total_loss = 0.0
            val_total_batches = 0
            
            with torch.no_grad():
                # Process each key in validation dict (similar to test_model)
                for key in val_input_dict.keys():
                    val_input_tensor = torch.from_numpy(val_input_dict[key].astype('float32'))
                    val_label_tensor = torch.from_numpy(val_label_dict[key].astype('float32'))
                    
                    val_set = torch.utils.data.TensorDataset(val_input_tensor, val_label_tensor)
                    val_loader = torch.utils.data.DataLoader(
                        val_set,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=torch.cuda.is_available(),
                        num_workers=4
                    )
                    
                    for val_inputs, val_labels in val_loader:
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        
                        # Generate model outputs (same logic as training)
                        if model_to_train == 'encoder_decoder':
                            val_model_outputs, val_decoder_outputs = model(val_inputs.float())
                        else:
                            val_model_outputs = model(val_inputs.float())
                            if pretrained_decoder:
                                val_decoder_outputs = decoder(val_model_outputs.float())
                            else:
                                val_decoder_outputs = None
                        
                        # Apply criterion (same logic as training and test_model)
                        if model_to_train == 'decoder':
                            # Check if temporal (similar to test_model logic)
                            temporal = len(val_labels.shape) == 5  # Assuming temporal if 5D
                            if temporal:
                                *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels)
                            else:
                                *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels[..., 15])
                        elif val_decoder_outputs is None:
                            *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels)
                        else:
                            *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels, val_decoder_outputs, val_inputs[..., 15])
                        
                        val_total_loss += val_total_loss_batch.item()
                        val_total_batches += 1

            # Calculate average validation loss across all keys and batches
            avg_val_loss = val_total_loss / val_total_batches if val_total_batches > 0 else 0.0

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
            if avg_val_loss is not None:
                print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)

        # Store losses in history
        history['total_loss'].append(avg_loss)
        # Only append validation loss when it's computed (every 5 epochs)
        if avg_val_loss is not None:
            history['validation_loss'].append(avg_val_loss)
        history['other_metrics'].append(avg_metrics)

    history['total_loss'] = np.asarray(history['total_loss'])
    history['validation_loss'] = np.asarray(history['validation_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    
    if display_plots:
        plot_train_losses_with_val(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history




def train_model_temporal_with_val_and_reg(input, label, val_input_dict, val_label_dict, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True, save_plots=False, model_name="", lambda_l1=None):
    """
    Train the model using the specified parameters and dataset.

    Arguments:
        input (numpy array): Input data to the model. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, 3, 112, 112, 32). Else, shape of (TR, mask_size).
        label (numpy array): Target labels for the input data. If model_to_train is 'encoder' or 'encoder_decoder', then shape of (TR, mask_size). Else, shape of (TR, 3, 112, 112, 32).
        val_input_dict (dict): Dictionary of validation input data. Keys are movie names, same structure as inputs_dict in test_model.
        val_label_dict (dict): Dictionary of validation labels. Keys are movie names, same structure as labels_dict in test_model.
        model (nn.Module): The neural network model to be trained.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        batch_size (int): Batch size for training.
        device (torch.device): Device to train the model on (CPU or GPU).
        save_model_as (str): Path to save the trained model.
        pretrained_decoder (str, optional): Path to a pretrained decoder model. Default is None.
        start_epoch (int, optional): Starting epoch number. Default is 1.
        start_loss (float, optional): Initial loss value. Default is None.
        model_to_train (str): Specifies which part of the model to train. Options are 'encoder', 'decoder', or 'encoder_decoder'.
        lambda_l1 (float, optional): Regularization strength for L1 regularization. Default is None.
        
    Returns:
        model (nn.Module): Trained model. The model is also stored at the specified path ('save_model_as')
        history (dict): Dictionary containing training loss history.
    """
    tic = time.time()

    print("input at beginning of function shape =", input.shape)
    print("label at beginning of function shape =", label.shape)
    print("val_input_dict keys =", list(val_input_dict.keys()))
    print("val_label_dict keys =", list(val_label_dict.keys()))

    # Create outputs directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs('outputs', exist_ok=True)

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
    
    # Prepare training data
    input = torch.from_numpy(input)
    label = torch.from_numpy(label)

    print("input after .from_numpy shape =", input.shape)
    print("label after .from_numpy shape =", label.shape)
    
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
        'validation_loss': [],  # Add validation loss tracking
        'other_metrics': [],
        'metrics_names': []
    }
    
    for epoch in range(start_epoch, num_epochs+1):
        # Training phase
        model.train()                                   
        running_loss = 0.0
        batch_count = 0
        epoch_metrics = None

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Determine the outputs based on the model configuration
            if model_to_train == 'encoder_decoder':
                model_outputs, decoder_outputs = model(inputs.float())
            else:
                #print("inputs shape to generate model_output =", inputs.shape)
                model_outputs = model(inputs.float())
                #print("model_outputs shape just after generating =", model_outputs.shape)
                if pretrained_decoder:
                    decoder_outputs = decoder(model_outputs.float())
                else:
                    decoder_outputs = None
        
            # Apply the appropriate criterion based on the presence of decoder outputs
            if model_to_train == 'decoder':
                #print("model_outputs shape =", model_outputs.shape)
                #print("labels shape =", labels.shape, "\n")
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            elif decoder_outputs is None:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels)
            else:
                *loss_metrics, total_loss, metrics_names = criterion(model_outputs, labels, decoder_outputs, inputs[..., 15])

            #apply regularization
            if lambda_l1 is not None:
                l1_loss = l1_regularization_loss(model, lambda_l1)
                total_loss = total_loss + l1_loss

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            batch_count += 1

            # DEBUG: Track running loss accumulation for first epoch
            if epoch == 1 and batch_count <= 10:
                print(f"  Batch {batch_count}: total_loss.item()={total_loss.item():.4f}, running_loss={running_loss:.4f}")

            # Clear some memory periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

            # Store metrics for averaging later
            if epoch_metrics is None:
                epoch_metrics = [metric.item() if hasattr(metric, 'item') else float(metric) for metric in loss_metrics]
            else:
                epoch_metrics = [epoch_metrics[j] + (metric.item() if hasattr(metric, 'item') else float(metric)) 
                                for j, metric in enumerate(loss_metrics)]

        # Calculate average training loss and metrics for the epoch
        avg_loss = running_loss / batch_count
        avg_metrics = [metric / batch_count for metric in epoch_metrics] if epoch_metrics else []

        # DEBUG: Print what's going into history
        if epoch <= 2:  # Debug first 2 epochs
            print(f"\nEPOCH {epoch} DEBUG:")
            print(f"  running_loss: {running_loss:.4f}")
            print(f"  batch_count: {batch_count}")
            print(f"  avg_loss: {avg_loss:.4f}")
            print(f"  avg_metrics: {avg_metrics}")
            print(f"  About to append avg_loss={avg_loss:.4f} to history")

        # Validation phase (run every 5 epochs)
        avg_val_loss = None
        if epoch % 5 == 0:
            model.eval()
            val_total_loss = 0.0
            val_total_batches = 0
            
            with torch.no_grad():
                # Process each key in validation dict (similar to test_model)
                for key in val_input_dict.keys():
                    val_input_tensor = torch.from_numpy(val_input_dict[key].astype('float32'))
                    val_label_tensor = torch.from_numpy(val_label_dict[key].astype('float32'))
                    
                    val_set = torch.utils.data.TensorDataset(val_input_tensor, val_label_tensor)
                    val_loader = torch.utils.data.DataLoader(
                        val_set,
                        batch_size=8,  # Reduced batch size for validation to save memory
                        shuffle=False,
                        pin_memory=torch.cuda.is_available(),
                        num_workers=2  # Reduced workers to save memory
                    )
                    
                    for val_inputs, val_labels in val_loader:
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        
                        # Generate model outputs (same logic as training)
                        if model_to_train == 'encoder_decoder':
                            val_model_outputs, val_decoder_outputs = model(val_inputs.float())
                        else:
                            val_model_outputs = model(val_inputs.float())
                            if pretrained_decoder:
                                val_decoder_outputs = decoder(val_model_outputs.float())
                            else:
                                val_decoder_outputs = None
                        
                        # Apply criterion (same logic as training and test_model)
                        if model_to_train == 'decoder':
                            # Check if temporal (similar to test_model logic)
                            temporal = len(val_labels.shape) == 5  # Assuming temporal if 5D
                            if temporal:
                                *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels)
                            else:
                                *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels[..., 15])
                        elif val_decoder_outputs is None:
                            *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels)
                        else:
                            *val_loss_metrics, val_total_loss_batch, val_metrics_names = criterion(val_model_outputs, val_labels, val_decoder_outputs, val_inputs[..., 15])
                        
                        val_total_loss += val_total_loss_batch.item()
                        val_total_batches += 1

            # Calculate average validation loss across all keys and batches
            avg_val_loss = val_total_loss / val_total_batches if val_total_batches > 0 else 0.0
            
            # Clear GPU memory after validation
            torch.cuda.empty_cache()

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
            if avg_val_loss is not None:
                print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss)

        # Store losses in history
        history['total_loss'].append(avg_loss)
        # Only append validation loss when it's computed (every 5 epochs)
        if avg_val_loss is not None:
            history['validation_loss'].append(avg_val_loss)
        history['other_metrics'].append(avg_metrics)

        # DEBUG: Show what was stored
        if epoch <= 2:
            print(f"  Stored in history: total_loss[-1]={history['total_loss'][-1]:.4f}")

    # DEBUG: Print final history before plotting
    print(f"\nFINAL HISTORY DEBUG:")
    print(f"history['total_loss']: {history['total_loss']}")
    print(f"history['validation_loss']: {history['validation_loss']}")
    print(f"First few other_metrics: {history['other_metrics'][:3]}")

    history['total_loss'] = np.asarray(history['total_loss'])
    history['validation_loss'] = np.asarray(history['validation_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    
    if display_plots:
        plot_train_losses_with_val(history, start_epoch, save_plots=save_plots, 
                         save_path=f'outputs/training_{model_name}.png' if save_plots else None)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history







#train_model_3(subject_folders, videos_folder, model, num_epochs, lr, criterion, optimizer, batch_size, 
#                device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, 
#                model_to_train=None, display_plots=True, save_plots=False, model_name="",
#                train_fraction=0.8, test_video=None, checkpoint_every=5):




