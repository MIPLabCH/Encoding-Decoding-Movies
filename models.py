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
from visualisation import plot_train_losses, plot_metrics, plot_decoder_predictions, plot_saliency_distribution, one_sample_permutation_test


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
        x = Resize([224,224])(x)
        x = normalize(x)
        output1 = self.block1(x)
        output2 = self.block2(output1)
        output3 = self.block3(output2)
        output4 = self.block4(output3)
        output5 = self.block5(output4)
        
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


def train_model(input, label, model, num_epochs, lr, criterion, optimizer, batch_size, device, save_model_as, pretrained_decoder=None, start_epoch=1, start_loss=None, model_to_train=None, display_plots=True):
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

        if epoch % 5 == 0 and display_plots:  # Every 5 epochs, print status
            print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {running_loss / (5*i):.4f}")
            running_loss = 0.0
            save_checkpoint(model, optimizer, epoch+1, total_loss)

        history['total_loss'].append(total_loss.item())
        history['other_metrics'].append(loss_metrics)  # Store other metrics for visualization

    history['total_loss'] = np.asarray(history['total_loss'])
    history['other_metrics'] = np.asarray(history['other_metrics'])
    history['metrics_names'] = metrics_names
    torch.save(model.state_dict(), save_model_as)
    if display_plots:
        plot_train_losses(history, start_epoch)

    print("Training completed. Total time: {:.2f} minutes".format((time.time() - tic) / 60))
    print('---')
    return model, history

def save_checkpoint(model, optimizer, epoch, loss):
    """
    Save the model checkpoint.

    Arguments:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
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


def test_model(inputs_dict, labels_dict, model, criterion, device, pretrained_decoder=None, model_to_test=None, statistical_testing = False, display_plots = True):
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
            plot_metrics(labels, encoded, key, plot_TR = False, performance_dict = None, display_plots=display_plots)

    if model_to_test != 'decoder':
        all_encoded = results['encoder_predictions']
        all_labels = labels_dict
        results['test_performance'] = plot_metrics(all_labels, all_encoded, 'all', plot_TR = True, performance_dict = results['test_performance'], display_plots = display_plots)
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
            results['test_performance'] = plot_decoder_predictions(results['decoder_predictions'], labels_dict, results['test_performance'], display_plots)
        else:
            results['test_performance'] = plot_decoder_predictions(results['decoder_predictions'], inputs_dict, results['test_performance'], display_plots)
        
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