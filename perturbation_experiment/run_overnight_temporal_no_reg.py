import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

#from dataset import *
from Video_dataset.models_new_2 import *
from Video_dataset.visualisation_new_2 import *
import torch

from Video_dataset.dataset_new import *
from Video_dataset.models_new_2 import *
from Video_dataset.visualisation_new_2 import *
from perturbation import *


### data loading ###

dataset_ID = 6661 # ID of a specific dataset. 6661 refer to preprocessed data with a mask of shape (4609,). 6660 refers to preprocessed data with a mask of shape (15364,)
mask_size = 4609 # number of voxels in the preprocessed fMRI data. either 4609 or 15364
trainset, valset, testset = get_dataset(dataset_ID, mask_size) # data are loaded into dictionaries



import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        #return total_psim.item(), total_ssim.item(), total_tv.item(), total_loss, metrics_names
        #return total_psim, total_ssim, total_tv, total_loss, metrics_names
        return total_psim.item(), total_ssim.item(), total_tv.item(), total_loss.item(), metrics_names
    



class Temporal_D_Loss_Debug(nn.Module):
    def __init__(self):
        super(Temporal_D_Loss_Debug, self).__init__()
        # Keep your existing loss components
        self.beta = 0.35
        self.gamma = 0.35
        self.delta = 0.30
        self.tv = TotalVariation()
        self.vgg16_features = VGG16Features().eval()
        self.batch_count = 0  # Add counter for debugging
        
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
        
        # DEBUG: Print first few batches of first epoch
        self.batch_count += 1
        if self.batch_count <= 5:  # First 5 batches
            print(f"\nBatch {self.batch_count} DEBUG:")
            print(f"  total_psim (tensor): {total_psim.item():.4f}")
            print(f"  total_ssim (tensor): {total_ssim.item():.4f}")
            print(f"  total_tv (tensor): {total_tv.item():.4f}")
            print(f"  Weights: beta={self.beta}, gamma={self.gamma}, delta={self.delta}")
            
            # Compute total loss step by step
            weighted_psim = self.beta * total_psim
            weighted_ssim = self.gamma * total_ssim
            weighted_tv = self.delta * total_tv
            
            print(f"  Weighted psim: {weighted_psim.item():.4f}")
            print(f"  Weighted ssim: {weighted_ssim.item():.4f}")
            print(f"  Weighted tv: {weighted_tv.item():.4f}")
            print(f"  Sum: {(weighted_psim + weighted_ssim + weighted_tv).item():.4f}")
        
        # Compute final loss with same weights
        total_loss = self.beta * total_psim + self.gamma * total_ssim + self.delta * total_tv
        
        if self.batch_count <= 5:
            print(f"  Final total_loss: {total_loss.item():.4f}")
        
        metrics_names = ['perc_sim', 'struct_sim', 'tv_loss', 'decoder_loss']
        return total_psim.item(), total_ssim.item(), total_tv.item(), total_loss, metrics_names


    

def prepare_temporal_data(fmri_data, all_frames, window_size=5):
    """
    Prepare temporal data with overlapping windows of TRs and their corresponding middle frames
    
    Args:
        fmri_data: Original fMRI data, shape (n_trs, mask_size)
        all_frames: All video frames, organized by TR, shape (n_trs, 3, 112, 112, 32)
        window_size: Number of consecutive TRs to use
    
    Returns:
        tr_windows: Windows of consecutive TRs
        frame_targets: Middle frames for each TR in the windows
    """
    n_trs = fmri_data.shape[0]
    tr_windows = []
    frame_targets = []

    print("preparing data...")
    
    # Create sliding windows of TRs
    for i in range(n_trs - window_size + 1):
        # Get window of TRs
        tr_window = fmri_data[i:i+window_size]
        tr_windows.append(tr_window)
        
        # Get middle frame for each TR in the window
        frames_for_window = []
        for j in range(window_size):
            # Get middle frame (assuming 32 frames per TR)
            middle_frame_idx = 15  # Middle of 32 frames (0-indexed, so 15 is the 16th frame)
            middle_frame = all_frames[i+j, :, :, :, middle_frame_idx]  # CORRECTED INDEXING
            frames_for_window.append(middle_frame)
        
        frame_targets.append(np.stack(frames_for_window))
    
    return np.array(tr_windows), np.array(frame_targets)

# Load dataset
dataset_ID = 6661
mask_size = 4609
trainset, valset, testset = get_dataset(dataset_ID, mask_size)

print_dict_tree(trainset)

### Prepare Temporal Data ###
window_size = 3  # Number of consecutive TRs to use

# Process training data to create temporal windows
train_input_temporal, train_label_temporal = prepare_temporal_data(
    trainset['fMRIs'], 
    trainset['videos'], 
    window_size=window_size
)

print(f"Original input shape: {trainset['fMRIs'].shape}")
print(f"Temporal input shape: {train_input_temporal.shape}")
print(f"Temporal label shape: {train_label_temporal.shape}")

### Model Training ###

# Training parameters
model = TemporalDecoder(mask_size, n_time_steps=window_size)
num_epochs = 352
lr = 1e-4
criterion = Temporal_D_Loss_Debug()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 16
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
save_model_as = '/media/RCPNAS/MIP/Michael/students_work/rodrigo/temporal_decoder_no_reg_' + str(mask_size) + '_' + str(num_epochs) + '_TRwindow' + str(window_size)
pretrained_decoder = None
start_epoch = 1
start_loss = None
model_to_train = 'decoder'
display_plots = True
save_plots = True
model_name = 'temporal_decoder_no_reg_' + str(mask_size) + '_' + str(num_epochs) + '_TRwindow' + str(window_size)

# Training loop
print("\n\nnew fmri shape:", train_input_temporal.shape)
print("new videos shape:", train_label_temporal.shape, "\n\n")

import time
#time.sleep(10)

#model, history = train_model_temporal(
#    train_input_temporal, train_label_temporal, 
#    model, num_epochs, lr, criterion, optimizer, 
#    batch_size, device, save_model_as, pretrained_decoder, 
#    start_epoch, start_loss, model_to_train, 
#    display_plots, save_plots, model_name
#)


def prepare_temporal_data_by_movie(fmri_data_dict, video_data_dict, window_size=3):
    """
    Prepare temporal data with overlapping windows of TRs and their corresponding middle frames,
    separately for each movie.
    
    Args:
        fmri_data_dict: Dictionary of fMRI data per movie, where keys are movie names and 
                        values have shape (n_trs, mask_size)
        video_data_dict: Dictionary of video frames per movie, where keys are movie names and 
                         values have shape (n_trs, 3, 112, 112, 32)
        window_size: Number of consecutive TRs to use
    
    Returns:
        tr_windows_dict: Dictionary of windows of consecutive TRs per movie
        frame_targets_dict: Dictionary of middle frames for each TR in the windows per movie
    """
    tr_windows_dict = {}
    frame_targets_dict = {}
    
    # Process each movie separately
    for movie_name in fmri_data_dict.keys():
        fmri_data = fmri_data_dict[movie_name]
        all_frames = video_data_dict[movie_name]
        
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
                middle_frame_idx = 15  # Middle of 32 frames (0-indexed, so 15 is the 16th frame)
                middle_frame = all_frames[i+j, :, :, :, middle_frame_idx]
                frames_for_window.append(middle_frame)
            
            frame_targets.append(np.stack(frames_for_window))
        
        # Store results for this movie
        tr_windows_dict[movie_name] = np.array(tr_windows)
        frame_targets_dict[movie_name] = np.array(frame_targets)
    
    return tr_windows_dict, frame_targets_dict


window_size = 3

valset3 = {
    "fMRIs": {},
    "videos": {}
}

valset3['fMRIs'], valset3['videos'] = prepare_temporal_data_by_movie(
    valset['fMRIs'], 
    valset['videos'], 
    window_size=window_size
)



model, history = train_model_temporal_with_val_and_reg(
    train_input_temporal, train_label_temporal, 
    valset3['fMRIs'], valset3['videos'],
    model, num_epochs, lr, criterion, optimizer, 
    batch_size, device, save_model_as, pretrained_decoder, 
    start_epoch, start_loss, model_to_train, 
    display_plots, save_plots, model_name, lambda_l1=0
)

print_dict_tree(history)
