import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from .conditional_unet1d import ConditionalUnet1D
from .gaussion_diffusion import D2MP_OB, VarianceSchedule

import sys
sys.path.append('../')

from utils.utils import *


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.LeakyReLU(0.1)
        # self.scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        # return self.activation(out + residual * self.scale)
        return self.activation(out + residual)
        

class DDTracking(nn.Module):
    def __init__(self, input_size, num_layers, num_cells, cell_type, dropout_rate, target_dim, loss_type, previous_dire=0, look_ahead=False, vox_step=0.5):
        super().__init__()

        self.target_dim = target_dim
        self.previous_dire = previous_dire

        self.cell_type = cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        
        self.cnn_out_dim = 192
        self.mlp_out_dim = 512

        self.look_ahead = look_ahead
        self.vox_step = vox_step

        self.local_cond_dim = 0
        if self.previous_dire > 0:
            self.local_cond_dim += 3 * self.previous_dire
         
        if self.look_ahead:
            self.local_cond_dim += self.mlp_out_dim

        if self.local_cond_dim == 0:
            self.local_cond_dim = None

        self.rnn = rnn_cls(
            input_size=num_cells,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        # self.rnn_layer_activation = nn.ReLU()

        self.net = ConditionalUnet1D(self.target_dim, local_cond_dim=self.local_cond_dim, global_cond_dim=num_cells)
        self.loss_type = loss_type
        self.diffusion = D2MP_OB(
            loss_type=self.loss_type,
            net=self.net,
            var_sched=VarianceSchedule(num_steps=100, beta_T=5e-2, mode='linear')
        )

        self.conv_layers1 = nn.Sequential(
            nn.Conv3d(in_channels=input_size, out_channels=self.cnn_out_dim, kernel_size=3, padding=0),
        )
        # kaiming initialization
        for m in self.conv_layers1.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.linear_proj1 = nn.Sequential(nn.Linear(self.cnn_out_dim, num_cells), *[self.block(num_cells, num_cells) for _ in range(2)])

        if self.look_ahead:
            self.conv_layers2 = nn.Sequential(
                nn.Conv3d(in_channels=input_size, out_channels=self.cnn_out_dim, kernel_size=3, padding=0),
            )
            self.linear_proj2 = nn.Sequential(nn.Linear(self.cnn_out_dim, self.mlp_out_dim), *[self.block(self.mlp_out_dim, self.mlp_out_dim) for _ in range(2)])
            
            # kaiming initialization
            for m in self.conv_layers2.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        self.offset = torch.as_tensor([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
        
    def block(self, in_features, out_features):
        if in_features == out_features:
            return ResBlock(in_features)
        else:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LeakyReLU(0.1)
            )

    def forward(self, volume, input_point, target, length):
        """
        Forward pass of the model.

        Args:
            volume (Tensor): Input volume of shape (B, D, H, W, C).
            input_point (Tensor): Input positions of shape (B, S, 3).
            target (Tensor): Target directions of shape (B, S, target_dim).
            length (List[int]): Lengths of sequences in the batch.

        Returns:
            loss (Tensor): Computed loss.
        """
        device = volume.device
        B, S, _ = input_point.shape
        volume_dim = volume.shape

        max_edge = torch.tensor(
            [volume_dim[1]-1, volume_dim[2]-1, volume_dim[3]-1],
            dtype=torch.float32,
            device=device
        )
        
        # Step 2: Process input points and sample from volume
        input_point_pack = rnn_utils.pack_padded_sequence(input_point, length, batch_first=True)
        input_point_dats = input_point_pack.data
        
        sampled_5d = interpolate_volume_in_neighborhood_torch(
            volume.permute(0, 4, 1, 2, 3),
            input_point_dats,
            max_edge,
            self.offset
        ).view(-1, 3, 3, 3, volume_dim[-1]).permute(0, 4, 1, 2, 3)
        sample_after_conv = self.conv_layers1(sampled_5d).view(-1, self.cnn_out_dim)
        sample_after_conv = self.linear_proj1(sample_after_conv)

        rnn_input = rnn_utils.PackedSequence(
            data=sample_after_conv,
            batch_sizes=input_point_pack.batch_sizes,
            sorted_indices=input_point_pack.sorted_indices,
            unsorted_indices=input_point_pack.unsorted_indices
        )
        
        global_feat_packed, _ = self.rnn(rnn_input, None)
        # global_feat = self.rnn_layer_activation(global_feat_packed.data)
        global_feat = global_feat_packed.data
        
        # Step 3: Local features - Look ahead and previous directions
        local_feat = []
        if self.look_ahead:
            temp_values = self.conv_layers2(sampled_5d).view(-1, self.cnn_out_dim)
            temp_values = self.linear_proj2(temp_values)
            local_feat.append(temp_values)

        # If previous_dire is enabled, append previous directions
        if self.previous_dire > 0:
            for i in range(self.previous_dire):
                zero_pad = torch.zeros((B, i+1, self.target_dim), device=device)
                prev_dire_seq = torch.cat([zero_pad, target[:, :-(1+i), :]], dim=1)
                prev_dire_pack = rnn_utils.pack_padded_sequence(prev_dire_seq, length, batch_first=True)
                prev_dire_data = prev_dire_pack.data
                local_feat.append(prev_dire_data)

        # Final local_feat
        if len(local_feat) > 0:
            local_feat = torch.cat(local_feat, dim=-1)  # (sum_len, local_cond_dim)
        else:
            local_feat = None
        
        # Step 4: Compute loss using the diffusion process
        target_pack = rnn_utils.pack_padded_sequence(
            target, 
            length, 
            batch_first=True
        )

        loss = self.diffusion(
            x_0=target_pack.data, 
            global_feat=global_feat,
            local_feat=local_feat
        )
        return loss
    
    def _apply_rnn_layers(self, volume, model_input, hidden_state):
        device = volume.device
        B, S, _ = model_input.shape
        volume_dim = volume.shape

        max_edge = torch.tensor(
            [volume_dim[1]-1, volume_dim[2]-1, volume_dim[3]-1],
            dtype=torch.float32,
            device=device
        )
        
        pos_input = model_input[..., :3]

        sampled_5d = interpolate_volume_in_neighborhood_torch(volume.permute(0, 4, 1, 2, 3), pos_input.view(-1, self.target_dim), max_edge, self.offset).view(-1, 3, 3, 3, volume_dim[-1]).permute(0, 4, 1, 2, 3)
        sample_after_conv = self.conv_layers1(sampled_5d).view(-1, self.cnn_out_dim)
        sample_after_conv = self.linear_proj1(sample_after_conv).view(B, S, -1)

        rnn_outputs, hidden_state = self.rnn(sample_after_conv, hidden_state)
        # rnn_outputs = self.rnn_layer_activation(rnn_outputs)

        return rnn_outputs, hidden_state
    
    def generate(self, volume, model_input, hidden_state):
        """
        Args:
            rnn_input: Input tensor of shape (B, D, H, W, C) -> (n, 3, 3, 3, 45)
            hidden_state: Hidden state for RNN
        Returns:
            new_samples, state
        """
        # Step 1: Apply RNN layers to the input
        device = volume.device
        B, S, _ = model_input.shape
        volume_dim = volume.shape

        max_edge = torch.tensor(
            [volume_dim[1]-1, volume_dim[2]-1, volume_dim[3]-1],
            dtype=torch.float32,
            device=device
        )
        
        pos_input = model_input[..., :3]

        sampled_5d = interpolate_volume_in_neighborhood_torch(volume.permute(0, 4, 1, 2, 3), pos_input.view(-1, self.target_dim), max_edge, self.offset).view(-1, 3, 3, 3, volume_dim[-1]).permute(0, 4, 1, 2, 3)
        sample_after_conv = self.conv_layers1(sampled_5d).view(-1, self.cnn_out_dim)
        sample_after_conv = self.linear_proj1(sample_after_conv).view(B, S, -1)

        rnn_outputs, hidden_state = self.rnn(sample_after_conv, hidden_state)
        
        # Get the global feature (last output of RNN)
        global_feat = rnn_outputs[:, -1, :] # (B, num_cells)
        
        local_feat = []

        # Step 2: If look_ahead is enabled, process ahead features
        if self.look_ahead:
            temps_values = self.conv_layers2(sampled_5d).view(B, S, -1)
            temps_values = self.linear_proj2(temps_values[:, -1, :])
            local_feat.append(temps_values)  # (B, cnn_out_dim)
        
        # Step 3: If previous_dire is enabled, append previous directions
        if self.previous_dire > 0:
            previous_dire_raw = model_input[:, -1, -self.target_dim*self.previous_dire:] # (B, previous_dire*target_dim)
            local_feat.append(previous_dire_raw)
        
        # Concatenate the local features if any
        if len(local_feat) > 0:
            local_feat = torch.cat(local_feat, dim=-1)
        else:
            local_feat = None

        # Step 4: Generate new samples using diffusion
        new_samples = self.diffusion.sample(
            global_feat=global_feat,  # (B, num_cells)
            local_feat=local_feat,
            bestof=True,
            point_dim=self.target_dim
        )  # (B, target_dim)

        return new_samples, hidden_state
    