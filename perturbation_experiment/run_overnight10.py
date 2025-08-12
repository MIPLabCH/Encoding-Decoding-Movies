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

class LSTMDecoderShared(nn.Module):
    def __init__(self, mask_size, n_time_steps=3, hidden_size=512, use_dropout=True):
        """
        LSTM Decoder where the CNN decoder weights are shared across time steps.
        This reduces overfitting by having fewer parameters.
        """
        super(LSTMDecoderShared, self).__init__()
        
        self.n_time_steps = n_time_steps
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=mask_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2 if use_dropout else 0.0,
            bidirectional=False
        )
        
        # Dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        
        # Single CNN decoder (shared across all time steps)
        self.cnn_decoder = nn.Sequential(
            # FC to spatial features
            nn.Linear(hidden_size, 14*14*48),
            nn.ReLU(),
            
            # Reshape happens in forward()
            
            # Conv layers (exactly your architecture)
            nn.ConvTranspose2d(48, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(48),
            
            nn.ConvTranspose2d(48, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(48),
            
            nn.ConvTranspose2d(48, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(48),
            
            nn.ConvTranspose2d(48, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.BatchNorm2d(3)
        )
        
        self.use_dropout = use_dropout
    
    def forward(self, x):
        batch_size, seq_len, mask_size = x.shape
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Flatten temporal and batch dimensions
        lstm_flat = lstm_out.reshape(-1, lstm_out.shape[-1])
        #lstm_flat = lstm_out.view(-1, lstm_out.shape[-1])  # (batch*seq, hidden_size)
        
        if self.use_dropout and self.training:
            lstm_flat = self.dropout(lstm_flat)
        
        # Process through CNN decoder
        # First FC layer
        cnn_input = self.cnn_decoder[0](lstm_flat)  # FC layer
        cnn_input = self.cnn_decoder[1](cnn_input)  # ReLU
        
        # Reshape for conv layers
        cnn_input = cnn_input.view(-1, 48, 14, 14)
        
        # Pass through remaining CNN layers
        output_flat = cnn_input
        for layer in self.cnn_decoder[2:]:  # Skip first FC and ReLU
            output_flat = layer(output_flat)
        
        # Reshape back to temporal format
        output = output_flat.view(batch_size, seq_len, 3, 112, 112)
        
        return output

    

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





class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        """
        Temporal attention module to focus on important time steps.
        """
        super(TemporalAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
            
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # Softmax over time dimension
        )
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, features)
        batch_size, n_time_steps, features = x.shape
        
        # Compute attention weights for each time step
        attn_weights = self.attention(x)  # (batch_size, n_time_steps, 1)
        
        # Apply attention weights
        attended = x * attn_weights  # Broadcast multiplication
        
        return attended, attn_weights.squeeze(-1)  # Return both attended features and weights


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        """
        Spatial attention module to focus on important spatial regions.
        """
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv3d(channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, channels, time, height, width)
        attention_map = self.conv1(x)
        attention_map = F.relu(attention_map)
        attention_map = self.conv2(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply spatial attention
        attended = x * attention_map
        
        return attended, attention_map


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        """
        Multi-head attention over temporal dimension.
        """
        super(MultiHeadTemporalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, features)
        
        # Self-attention over temporal dimension
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        
        # Residual connection and normalization
        output = self.norm(x + self.dropout(attn_output))
        
        return output, attn_weights


class AttentionTemporalDecoder(nn.Module):
    def __init__(self, mask_size, n_time_steps=3, attention_type='temporal'):
        """
        Temporal Decoder enhanced with attention mechanisms.
        
        Arguments:
            mask_size (int): The size of the input layer.
            n_time_steps (int): Number of consecutive TRs to process together.
            attention_type (str): Type of attention - 'temporal', 'spatial', 'both', 'multihead'
        """
        super(AttentionTemporalDecoder, self).__init__()
        
        self.n_time_steps = n_time_steps
        self.attention_type = attention_type
        
        # Original temporal decoder components
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        # Temporal attention after FC layer
        if attention_type in ['temporal', 'both', 'multihead']:
            if attention_type == 'multihead':
                self.temporal_attention = MultiHeadTemporalAttention(
                    input_dim=14*14*48, 
                    num_heads=8, 
                    dropout=0.1
                )
            else:
                self.temporal_attention = TemporalAttention(14*14*48)
        
        # 3D convolutional layers (same as original)
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn1 = nn.BatchNorm3d(48)
        
        # Spatial attention after first conv block
        if attention_type in ['spatial', 'both']:
            self.spatial_attention1 = SpatialAttention(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn2 = nn.BatchNorm3d(48)
        
        # Spatial attention after second conv block
        if attention_type in ['spatial', 'both']:
            self.spatial_attention2 = SpatialAttention(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn3 = nn.BatchNorm3d(48)
        
        # Final layer maintains temporal dimension
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn4 = nn.BatchNorm3d(3)
        
        # Store attention weights for visualization
        self.last_attention_weights = {}
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, mask_size)
        batch_size, n_time_steps, mask_size = x.shape
        
        # Process each TR independently first
        x_reshaped = x.reshape(-1, mask_size)
        x_fc = self.fc(x_reshaped)
        x_fc = F.relu(x_fc)
        
        # Reshape back to temporal format for attention
        x_temporal = x_fc.view(batch_size, n_time_steps, -1)  # (batch_size, n_time_steps, 14*14*48)
        
        # Apply temporal attention if enabled
        if self.attention_type in ['temporal', 'both', 'multihead']:
            x_temporal, temporal_attn_weights = self.temporal_attention(x_temporal)
            self.last_attention_weights['temporal'] = temporal_attn_weights
            
        # Reshape to 3D volume with temporal dimension
        x = x_temporal.view(batch_size, n_time_steps, 48, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, 48, n_time_steps, 14, 14)
        
        # First 3D conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.up1(x)
        x = self.bn1(x)
        
        # Apply spatial attention if enabled
        if self.attention_type in ['spatial', 'both']:
            x, spatial_attn1 = self.spatial_attention1(x)
            self.last_attention_weights['spatial_1'] = spatial_attn1
        
        # Second 3D conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.up2(x)
        x = self.bn2(x)
        
        # Apply spatial attention if enabled
        if self.attention_type in ['spatial', 'both']:
            x, spatial_attn2 = self.spatial_attention2(x)
            self.last_attention_weights['spatial_2'] = spatial_attn2
        
        # Third 3D conv block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.up3(x)
        x = self.bn3(x)
        
        # Final layer
        x = self.conv4(x)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        # Rearrange to put channels after temporal dimension
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, n_time_steps, 3, 112, 112)
        
        return x
    
    def get_attention_weights(self):
        """Return the last computed attention weights for visualization."""
        return self.last_attention_weights


# Lightweight version with just temporal attention
class TemporalDecoderWithAttention(nn.Module):
    def __init__(self, mask_size, n_time_steps=3, use_multihead=False):
        """
        Simplified version with just temporal attention added to your original decoder.
        """
        super(TemporalDecoderWithAttention, self).__init__()
        
        self.n_time_steps = n_time_steps
        self.use_multihead = use_multihead
        
        # Original components
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        # Temporal attention
        if use_multihead:
            self.temporal_attention = MultiHeadTemporalAttention(
                input_dim=14*14*48,
                num_heads=8,
                dropout=0.1
            )
        else:
            self.temporal_attention = TemporalAttention(14*14*48)
        
        # Original 3D conv layers (unchanged)
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn1 = nn.BatchNorm3d(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn3 = nn.BatchNorm3d(48)
        
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn4 = nn.BatchNorm3d(3)
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, mask_size)
        batch_size, n_time_steps, mask_size = x.shape
        
        # Process each TR independently first
        x_reshaped = x.reshape(-1, mask_size)
        x_fc = self.fc(x_reshaped)
        x_fc = F.relu(x_fc)
        
        # Reshape for temporal attention
        x_temporal = x_fc.view(batch_size, n_time_steps, -1)
        
        # Apply temporal attention
        x_attended, attention_weights = self.temporal_attention(x_temporal)
        
        # Continue with original 3D conv processing
        x = x_attended.view(batch_size, n_time_steps, 48, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Original 3D conv layers
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
        
        x = self.conv4(x)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        return x


# Even simpler version - just add attention at the end
class TemporalDecoderSimpleAttention(nn.Module):
    def __init__(self, mask_size, n_time_steps=3):
        """
        Minimal modification: just add temporal attention at the input level.
        """
        super(TemporalDecoderSimpleAttention, self).__init__()
        
        self.n_time_steps = n_time_steps
        
        # Simple temporal attention on input fMRI features
        self.input_attention = TemporalAttention(mask_size)
        
        # Your exact original decoder (unchanged)
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn1 = nn.BatchNorm3d(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn3 = nn.BatchNorm3d(48)
        
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn4 = nn.BatchNorm3d(3)
        
    def forward(self, x):
        # Apply attention to input fMRI features
        x_attended, attention_weights = self.input_attention(x)
        
        # Rest is exactly your original forward pass
        batch_size, n_time_steps, mask_size = x_attended.shape
        
        x_reshaped = x_attended.reshape(-1, mask_size)
        x = self.fc(x_reshaped)
        x = F.relu(x)
        
        x = x.view(batch_size, n_time_steps, 48, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)
        
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
        
        x = self.conv4(x)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        return x










class SpatialBrainAttention(nn.Module):
    def __init__(self, mask_size, hidden_dim=256):
        """
        Attention over brain regions/voxels to focus on visual cortex areas.
        This makes more sense than temporal attention for brain decoding.
        """
        super(SpatialBrainAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(mask_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, mask_size),
            nn.Sigmoid()  # Attention weights for each voxel
        )
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, mask_size)
        batch_size, n_time_steps, mask_size = x.shape
        
        # Compute attention weights for each voxel (averaged across time)
        avg_activity = x.mean(dim=1)  # (batch_size, mask_size)
        voxel_weights = self.attention(avg_activity)  # (batch_size, mask_size)
        
        # Apply spatial attention to all time steps
        voxel_weights = voxel_weights.unsqueeze(1)  # (batch_size, 1, mask_size)
        attended = x * voxel_weights  # Broadcast across time dimension
        
        return attended, voxel_weights.squeeze(1)


class CrossTemporalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        """
        Cross-attention: Each output frame can attend to ALL input time steps.
        This is more appropriate for brain decoding.
        """
        super(CrossTemporalAttention, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query, key_value):
        # query: (batch_size, output_steps, feature_dim) - what we want to generate
        # key_value: (batch_size, input_steps, feature_dim) - what we have from brain
        
        attn_output, attn_weights = self.multihead_attn(query, key_value, key_value)
        output = self.norm(query + attn_output)
        
        return output, attn_weights


class BrainOptimizedTemporalDecoder(nn.Module):
    def __init__(self, mask_size, n_time_steps=3, use_spatial_attention=True, use_cross_attention=False):
        """
        Temporal decoder optimized for brain-to-video reconstruction.
        
        Arguments:
            mask_size (int): fMRI mask size
            n_time_steps (int): Number of time steps
            use_spatial_attention (bool): Focus on important brain regions
            use_cross_attention (bool): Let output frames attend to all input steps
        """
        super(BrainOptimizedTemporalDecoder, self).__init__()
        
        self.n_time_steps = n_time_steps
        self.use_spatial_attention = use_spatial_attention
        self.use_cross_attention = use_cross_attention
        
        # Spatial attention on brain regions (most important for brain decoding)
        if use_spatial_attention:
            self.brain_attention = SpatialBrainAttention(mask_size)
        
        # Original components
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        # Cross-temporal attention after FC (optional)
        if use_cross_attention:
            self.cross_attention = CrossTemporalAttention(14*14*48, num_heads=4)
        
        # Original 3D conv layers (unchanged - these work well!)
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn1 = nn.BatchNorm3d(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn3 = nn.BatchNorm3d(48)
        
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn4 = nn.BatchNorm3d(3)
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, mask_size)
        batch_size, n_time_steps, mask_size = x.shape
        
        # Step 1: Spatial attention on brain regions (focus on visual cortex)
        if self.use_spatial_attention:
            x, brain_attn_weights = self.brain_attention(x)
        
        # Step 2: Extract features from each time step
        x_reshaped = x.reshape(-1, mask_size)
        x_fc = self.fc(x_reshaped)
        x_fc = F.relu(x_fc)
        
        # Reshape back to temporal format
        x_temporal = x_fc.view(batch_size, n_time_steps, -1)
        
        # Step 3: Cross-temporal attention (optional)
        if self.use_cross_attention:
            # Each output frame can attend to all input time steps
            x_temporal, cross_attn_weights = self.cross_attention(x_temporal, x_temporal)
        
        # Step 4: Continue with proven 3D conv processing
        x = x_temporal.view(batch_size, n_time_steps, 48, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Original 3D conv layers (keep these - they work!)
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
        
        x = self.conv4(x)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        return x


class FeatureLevelTemporalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        """
        Attention AFTER feature extraction but BEFORE 3D convolutions.
        This decides how to best combine temporal features for reconstruction.
        """
        super(FeatureLevelTemporalAttention, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, n_time_steps, feature_dim)
        
        # Self-attention over temporal features
        attn_output, attn_weights = self.self_attention(x, x, x)
        
        # Residual connection
        output = self.norm(x + self.dropout(attn_output))
        
        return output, attn_weights


class TemporalDecoderFeatureAttention(nn.Module):
    def __init__(self, mask_size, n_time_steps=3, use_spatial_brain_attention=True):
        """
        Best of both worlds: 
        1. Spatial attention on brain regions
        2. Temporal attention on extracted features
        3. Keep proven 3D conv architecture
        """
        super(TemporalDecoderFeatureAttention, self).__init__()
        
        self.n_time_steps = n_time_steps
        self.use_spatial_brain_attention = use_spatial_brain_attention
        
        # Spatial attention on brain voxels
        if use_spatial_brain_attention:
            self.brain_attention = SpatialBrainAttention(mask_size)
        
        # Feature extraction
        self.fc = nn.Linear(mask_size, 14*14*48)
        
        # Temporal attention on extracted features
        self.feature_attention = FeatureLevelTemporalAttention(14*14*48, num_heads=8)
        
        # Your proven 3D conv architecture
        self.conv1 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn1 = nn.BatchNorm3d(48)
        
        self.conv2 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn2 = nn.BatchNorm3d(48)
        
        self.conv3 = nn.Conv3d(48, 48, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.bn3 = nn.BatchNorm3d(48)
        
        self.conv4 = nn.Conv3d(48, 3, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.bn4 = nn.BatchNorm3d(3)
        
    def forward(self, x):
        batch_size, n_time_steps, mask_size = x.shape
        
        # 1. Focus on important brain regions (visual cortex areas)
        if self.use_spatial_brain_attention:
            x, brain_attn = self.brain_attention(x)
        
        # 2. Extract features from brain activity
        x_reshaped = x.reshape(-1, mask_size)
        x_fc = self.fc(x_reshaped)
        x_fc = F.relu(x_fc)
        
        # 3. Reshape and apply temporal attention on features
        x_temporal = x_fc.view(batch_size, n_time_steps, -1)
        x_attended, temporal_attn = self.feature_attention(x_temporal)
        
        # 4. Continue with proven 3D processing
        x = x_attended.view(batch_size, n_time_steps, 48, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Your proven 3D conv layers
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
        
        x = self.conv4(x)
        x = torch.sigmoid(x)
        x = self.bn4(x)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        return x









    

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
#model= LSTMDecoderShared(mask_size, n_time_steps=window_size, use_dropout=True)
#model = TemporalDecoderWithAttention(mask_size, n_time_steps=window_size, 
#                                   use_multihead=True)
# This is the one with both attention mechanisms:
model = TemporalDecoderFeatureAttention(
    mask_size=mask_size, 
    n_time_steps=window_size, 
    use_spatial_brain_attention=True
)
#model = TemporalDecoder(mask_size, n_time_steps=window_size)
num_epochs = 352
lr = 1e-4
criterion = Temporal_D_Loss_Debug()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 16
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#save_model_as = '/media/RCPNAS/MIP/Michael/students_work/rodrigo/temporal_decoder_' + str(mask_size) + '_' + str(num_epochs) + '_TRwindow' + str(window_size)
save_model_as = '/media/RCPNAS/MIP/Michael/students_work/rodrigo/temporal+attention_' + str(mask_size) + '_' + str(num_epochs) + '_TRwindow' + str(window_size)
pretrained_decoder = None
start_epoch = 1
start_loss = None
model_to_train = 'decoder'
display_plots = True
save_plots = True
#model_name = 'temporal_decoder_' + str(mask_size) + '_' + str(num_epochs) + '_TRwindow' + str(window_size)
model_name = 'temporal+attention_' + str(mask_size) + '_' + str(num_epochs) + '_TRwindow' + str(window_size)


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
    display_plots, save_plots, model_name, lambda_l1=1e-4
)

print_dict_tree(history)
