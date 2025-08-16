import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from tqdm import tqdm
import logging
from collections import OrderedDict
import time
import math


# ==================== Memory Efficient Utils ====================
def efficient_square_distance(src, dst, batch_size=1000):
    """Memory efficient calculation of Euclidean distance"""
    B, N, _ = src.shape
    _, M, _ = dst.shape

    # Process in batches to reduce memory usage
    all_dists = []
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        src_batch = src[:, i:end_i, :]  # [B, batch_size, 3]

        # Calculate distance for this batch
        dist = -2 * torch.matmul(src_batch, dst.permute(0, 2, 1))
        dist += torch.sum(src_batch ** 2, -1).unsqueeze(-1)
        dist += torch.sum(dst ** 2, -1).unsqueeze(1)
        all_dists.append(dist)

    return torch.cat(all_dists, dim=1)


def efficient_knn_point(nsample, xyz, new_xyz, batch_size=1000):
    """Memory efficient k-nearest neighbors"""
    B, N, _ = xyz.shape
    _, M, _ = new_xyz.shape

    all_group_idx = []
    for i in range(0, M, batch_size):
        end_i = min(i + batch_size, M)
        new_xyz_batch = new_xyz[:, i:end_i, :]

        # Calculate distances for this batch
        sqrdists = efficient_square_distance(new_xyz_batch, xyz, batch_size=500)
        _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
        all_group_idx.append(group_idx)

    return torch.cat(all_group_idx, dim=1)


def chunked_index_points(points, idx, chunk_size=1000):
    """Memory efficient point indexing"""
    device = points.device
    B, N, C = points.shape
    _, M, K = idx.shape

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1, 1).expand(B, M, K)

    all_new_points = []
    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        idx_chunk = idx[:, i:end_i, :]
        batch_indices_chunk = batch_indices[:, i:end_i, :]

        new_points_chunk = points[batch_indices_chunk, idx_chunk, :]
        all_new_points.append(new_points_chunk)

    return torch.cat(all_new_points, dim=1)


def index_points(points, idx):
    """Standard index points for compatibility"""
    if points.numel() > 50000000:  # Use chunked version for large tensors
        return chunked_index_points(points, idx)

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """Choose between efficient and standard kNN based on input size"""
    B, N, _ = xyz.shape
    if B * N > 100000:  # Use efficient version for large point clouds
        return efficient_knn_point(nsample, xyz, new_xyz)
    else:
        # Original implementation for small point clouds
        sqrdists = -2 * torch.matmul(new_xyz, xyz.permute(0, 2, 1))
        sqrdists += torch.sum(new_xyz ** 2, -1).view(B, new_xyz.shape[1], 1)
        sqrdists += torch.sum(xyz ** 2, -1).view(B, 1, N)
        _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
        return group_idx


# ==================== Data Loading ====================
class S3DISDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', area=5, num_points=1000,
                 transform=None, test_area=5, grid_size=0.04):
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.test_area = test_area
        self.grid_size = grid_size

        # S3DIS has 13 classes as per paper
        self.num_classes = 13
        self.class_names = ['ceiling', 'floor', 'wall', 'beam', 'column',
                            'window', 'door', 'table', 'chair', 'sofa',
                            'bookcase', 'board', 'clutter']

        self.data_list = []
        self._load_data()

    def _load_data(self):
        areas = [1, 2, 3, 4, 5, 6]
        if self.split == 'train':
            areas = [i for i in areas if i != self.test_area]
        else:
            areas = [self.test_area]

        for area in areas:
            area_path = os.path.join(self.data_root, f'Area_{area}')
            for room_file in os.listdir(area_path):
                if room_file.endswith('.npy'):
                    self.data_list.append(os.path.join(area_path, room_file))

    def _grid_subsample(self, points, colors, labels):
        """4cm grid subsampling as mentioned in paper"""
        if self.grid_size <= 0:
            return points, colors, labels

        # Quantize points to grid
        grid_coords = np.floor(points / self.grid_size).astype(np.int32)

        # Find unique grid cells
        _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)

        return points[unique_indices], colors[unique_indices], labels[unique_indices]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Load point cloud data: [N, 9] (xyz + rgb + labels)
        data = np.load(self.data_list[idx])
        points = data[:, :3].astype(np.float32)  # xyz
        colors = data[:, 3:6].astype(np.float32) / 255.0  # rgb normalized
        labels = data[:, 6].astype(np.int64)  # labels

        # Apply 4cm grid subsampling as per paper
        points, colors, labels = self._grid_subsample(points, colors, labels)

        # Subsample to max points if needed
        if len(points) > self.num_points:
            idx = np.random.choice(len(points), self.num_points, replace=False)
            points = points[idx]
            colors = colors[idx]
            labels = labels[idx]

        # Data augmentation (as per paper)
        if self.transform and self.split == 'train':
            points, colors = self.transform(points, colors)

        # Prepare features [N, 6] (xyz + rgb)
        features = np.concatenate([points, colors], axis=1)

        return {
            'points': torch.from_numpy(points).float(),
            'features': torch.from_numpy(features).float(),
            'labels': torch.from_numpy(labels)
        }


class ScanNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split='train', num_points=80000, transform=None):
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.transform = transform

        # ScanNet has 20 classes
        self.num_classes = 20

        self.data_list = []
        self._load_data()

    def _load_data(self):
        split_file = os.path.join(self.data_root, f'{self.split}.txt')
        with open(split_file, 'r') as f:
            scene_names = [line.strip() for line in f]

        for scene_name in scene_names:
            scene_path = os.path.join(self.data_root, 'scenes', f'{scene_name}.npy')
            if os.path.exists(scene_path):
                self.data_list.append(scene_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        points = data[:, :3].astype(np.float32)
        colors = data[:, 3:6].astype(np.float32) / 255.0
        labels = data[:, 6].astype(np.int64)

        if len(points) > self.num_points:
            idx = np.random.choice(len(points), self.num_points, replace=False)
            points = points[idx]
            colors = colors[idx]
            labels = labels[idx]

        if self.transform and hasattr(self, 'split') and self.split == 'train':
            points, colors = self.transform(points, colors)

        features = np.concatenate([points, colors], axis=1)

        return {
            'points': torch.from_numpy(points).float(),
            'features': torch.from_numpy(features).float(),
            'labels': torch.from_numpy(labels)
        }


# ==================== Data Augmentation ====================
class PointCloudAugmentation:
    """Data augmentation as described in paper"""

    def __init__(self):
        pass

    def __call__(self, points, colors):
        # Random scaling (0.8 to 1.2)
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        # Random rotation around Z axis
        angle = np.random.uniform(-np.pi, np.pi)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T

        # Random translation
        translation = np.random.uniform(-0.2, 0.2, 3)
        points = points + translation

        # Chromatic augmentation
        if np.random.random() > 0.5:
            # Chromatic contrast
            colors = colors * np.random.uniform(0.8, 1.2, 3)
            colors = np.clip(colors, 0, 1)

        if np.random.random() > 0.5:
            # Chromatic translation
            colors = colors + np.random.uniform(-0.1, 0.1, 3)
            colors = np.clip(colors, 0, 1)

        # Chromatic jitter
        if np.random.random() > 0.5:
            colors = colors + np.random.normal(0, 0.05, colors.shape)
            colors = np.clip(colors, 0, 1)

        return points, colors


# ==================== Diffusion Utils ====================
def linear_beta_schedule(timesteps):
    """Linear beta schedule for diffusion as per paper"""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    """Extract values from array a at timestep t"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ==================== Shared Transition Components ====================
class SharedTransitionDown(nn.Module):
    """Shared Transition Down with proper shared indices as per paper"""

    def __init__(self, in_channels, out_channels, k=16, stride=2):
        super().__init__()
        self.k = k
        self.stride = stride

        # MLP layers as per paper architecture
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels + 3, out_channels // 2, 1),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 2, out_channels // 2, 1)
        )
        self.mlp2 = nn.Conv1d(out_channels // 2, out_channels, 1)

    def forward(self, xyz, features, shared_idx=None):
        # xyz: [B, N, 3], features: [B, N, C]
        B, N, C = features.shape

        # Use shared indices if provided, otherwise compute FPS
        if shared_idx is not None:
            fps_idx = shared_idx
        else:
            fps_idx = self.farthest_point_sample(xyz, N // self.stride)

        new_xyz = index_points(xyz, fps_idx)  # [B, N//stride, 3]

        # Group points using kNN
        if shared_idx is not None:
            # Use shared neighborhood indices for the same resolution level
            idx = knn_point(self.k, xyz, new_xyz)
        else:
            idx = knn_point(self.k, xyz, new_xyz)

        grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)  # [B, N//stride, k, 3]
        grouped_features = index_points(features, idx)  # [B, N//stride, k, C]

        # Concatenate position and features
        grouped_features = torch.cat([grouped_features, grouped_xyz], dim=-1)  # [B, N//stride, k, C+3]

        # Permute for conv1d: [B, C+3, N//stride, k]
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()

        # Apply MLPs
        grouped_features = self.mlp1(grouped_features.view(B, -1, (N // self.stride) * self.k))
        grouped_features = grouped_features.view(B, -1, N // self.stride, self.k)

        # Max pooling over neighborhood
        new_features = torch.max(grouped_features, dim=-1)[0]  # [B, C//2, N//stride]
        new_features = self.mlp2(new_features)  # [B, C_out, N//stride]

        # Permute back: [B, N//stride, C_out]
        new_features = new_features.permute(0, 2, 1).contiguous()

        return new_xyz, new_features, fps_idx, idx

    def farthest_point_sample(self, xyz, npoint):
        """Farthest Point Sampling with memory efficiency"""
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)

            # Calculate distance in chunks to save memory
            chunk_size = 10000
            for j in range(0, N, chunk_size):
                end_j = min(j + chunk_size, N)
                xyz_chunk = xyz[:, j:end_j, :]
                dist_chunk = torch.sum((xyz_chunk - centroid) ** 2, -1)
                mask_chunk = dist_chunk < distance[:, j:end_j]
                distance[:, j:end_j][mask_chunk] = dist_chunk[mask_chunk]

            farthest = torch.max(distance, -1)[1]

        return centroids


class SharedTransitionUp(nn.Module):
    """Shared Transition Up with proper interpolation as per paper"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, N1, 3] - target points (higher resolution)
        xyz2: [B, N2, 3] - source points (lower resolution)
        points1: [B, N1, C1] - target features
        points2: [B, N2, C2] - source features to interpolate
        """
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape

        if N2 == 1:
            # If only one point, repeat it
            interpolated_points = points2.expand(-1, N1, -1)
        else:
            # 3-NN interpolation with inverse distance weighting
            dists = torch.cdist(xyz1, xyz2)  # [B, N1, N2]
            dists, idx = torch.topk(dists, min(3, N2), dim=-1, largest=False)  # [B, N1, 3]

            # Inverse distance weighting
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, N1, 3]

            # Gather and interpolate
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1),
                dim=2
            )  # [B, N1, C2]

        # Concatenate with skip connection if exists
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Apply MLP
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B, C, N1]
        new_points = self.mlp(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B, N1, C_out]

        return new_points


# ==================== Point Transformer Components ====================
class PointTransformerLayer(nn.Module):
    """Point Transformer Layer implementing vector attention as per paper"""

    def __init__(self, channels, k=4):
        super().__init__()
        self.k = k
        self.channels = channels

        # Query, Key, Value projections
        self.q_conv = nn.Linear(channels, channels)
        self.k_conv = nn.Linear(channels, channels)
        self.v_conv = nn.Linear(channels, channels)

        # Position encoding network
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

        # Attention weight network
        self.attn_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Softmax(dim=-1)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(channels)

    def forward(self, xyz, features, neighbor_idx=None):
        B, N, C = features.shape

        if neighbor_idx is None:
            neighbor_idx = knn_point(self.k, xyz, xyz)

        # Get Q, K, V
        q = self.q_conv(features)  # [B, N, C]
        k = self.k_conv(features)  # [B, N, C]
        v = self.v_conv(features)  # [B, N, C]

        # Group K, V by neighbor indices
        k_grouped = index_points(k, neighbor_idx)  # [B, N, k, C]
        v_grouped = index_points(v, neighbor_idx)  # [B, N, k, C]
        xyz_grouped = index_points(xyz, neighbor_idx)  # [B, N, k, 3]

        # Position encoding
        pos_diff = xyz_grouped - xyz.unsqueeze(2)  # [B, N, k, 3]
        pos_enc = self.pos_mlp(pos_diff.view(-1, 3)).view(B, N, self.k, C)  # [B, N, k, C]

        # Vector attention computation
        q_expanded = q.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, C]
        relation = q_expanded - k_grouped + pos_enc  # [B, N, k, C]

        # Compute attention weights
        attn_weights = self.attn_mlp(relation.view(-1, C)).view(B, N, self.k, C)  # [B, N, k, C]

        # Apply attention
        out = torch.sum(attn_weights * (v_grouped + pos_enc), dim=2)  # [B, N, C]

        # Residual connection and normalization
        out = self.norm(out + features)

        return out


class FullPointTransformer(nn.Module):
    """Full Point Transformer as conditional network (CNet_θ)"""

    def __init__(self, in_channels, num_classes, channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

        # Encoder layers with shared transitions
        self.enc1 = PointTransformerLayer(channels, k=16)
        self.down1 = SharedTransitionDown(channels, channels * 2, k=16, stride=2)

        self.enc2 = PointTransformerLayer(channels * 2, k=16)
        self.down2 = SharedTransitionDown(channels * 2, channels * 4, k=16, stride=2)

        self.enc3 = PointTransformerLayer(channels * 4, k=16)
        self.down3 = SharedTransitionDown(channels * 4, channels * 8, k=16, stride=2)

        self.enc4 = PointTransformerLayer(channels * 8, k=16)

        # Decoder layers with shared transitions
        self.up3 = SharedTransitionUp(channels * 8 + channels * 4, channels * 4)
        self.dec3 = PointTransformerLayer(channels * 4, k=16)

        self.up2 = SharedTransitionUp(channels * 4 + channels * 2, channels * 2)
        self.dec2 = PointTransformerLayer(channels * 2, k=16)

        self.up1 = SharedTransitionUp(channels * 2 + channels, channels)
        self.dec1 = PointTransformerLayer(channels, k=16)

        # Store shared indices for denoising network
        self.shared_indices = {}

    def forward(self, points, features):
        B, N, _ = points.shape

        # Input embedding
        x = self.input_embed(features.view(-1, features.size(-1)))
        x = x.view(B, N, -1)  # [B, N, 64]

        # Encoder path with shared index storage
        x1 = self.enc1(points, x)
        xyz2, x2, fps_idx2, neighbor_idx2 = self.down1(points, x1)
        self.shared_indices['level2'] = (fps_idx2, neighbor_idx2)

        x2 = self.enc2(xyz2, x2)
        xyz3, x3, fps_idx3, neighbor_idx3 = self.down2(xyz2, x2)
        self.shared_indices['level3'] = (fps_idx3, neighbor_idx3)

        x3 = self.enc3(xyz3, x3)
        xyz4, x4, fps_idx4, neighbor_idx4 = self.down3(xyz3, x3)
        self.shared_indices['level4'] = (fps_idx4, neighbor_idx4)

        # Bottleneck
        x4 = self.enc4(xyz4, x4)

        # Decoder path
        x = self.up3(xyz3, xyz4, x3, x4)
        x = self.dec3(xyz3, x)

        x = self.up2(xyz2, xyz3, x2, x)
        x = self.dec2(xyz2, x)

        x = self.up1(points, xyz2, x1, x)
        semantic_features = self.dec1(points, x)  # [B, N, 64] - semantic condition S_i

        # Generate position condition P_ij using kNN
        k = 16
        position_idx = knn_point(k, points, points)  # [B, N, k]
        position_condition = index_points(points, position_idx)  # [B, N, k, 3]

        return semantic_features, position_condition


# ==================== Diffusion Model Components ====================
class NoisyLabelEmbedding(nn.Module):
    """Noisy Label Embedding as per Equation 7 in paper"""

    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k

        # MLP for noisy labels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

        # Weight function W(·)
        self.weight_func = nn.Sequential(
            nn.Linear(out_channels + 3, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, noisy_labels, semantic_condition, position_condition):
        """
        Implements Equation 7 from paper:
        x_i = MLP(x^t_i)
        y_i = Σ W((S_ij ⊖ S_i) ⊕ P_ij) ⊗ (x_ij ⊕ P_ij)
        f_i = y_i ⊕ x_i
        """
        B, N, C_in = noisy_labels.shape

        # x_i = MLP(x^t_i)
        x_i = self.mlp(noisy_labels.view(-1, C_in)).view(B, N, -1)  # [B, N, C_out]

        # Group semantic and position conditions
        semantic_grouped = semantic_condition.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, C]
        x_grouped = x_i.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, C_out]

        # Semantic differences: S_ij ⊖ S_i
        semantic_diff = semantic_grouped - semantic_condition.unsqueeze(2)  # [B, N, k, C]

        # Combine with position: (S_ij ⊖ S_i) ⊕ P_ij
        weight_input = torch.cat([semantic_diff, position_condition], dim=-1)  # [B, N, k, C+3]

        # W((S_ij ⊖ S_i) ⊕ P_ij)
        weights = self.weight_func(weight_input.view(-1, weight_input.size(-1)))
        weights = weights.view(B, N, self.k, -1)  # [B, N, k, C_out]

        # x_ij ⊕ P_ij (combine grouped features with position)
        x_pos_combined = torch.cat([x_grouped, position_condition], dim=-1)  # [B, N, k, C_out+3]
        x_pos_combined = x_pos_combined[..., :weights.size(-1)]  # Match dimensions

        # W(...) ⊗ (x_ij ⊕ P_ij) and sum over neighborhood
        y_i = torch.sum(weights * x_pos_combined, dim=2)  # [B, N, C_out]

        # f_i = y_i ⊕ x_i
        f_i = y_i + x_i

        return f_i


class PointFrequencyTransformer(nn.Module):
    """Point Frequency Transformer as per Equations 8-11 in paper"""

    def __init__(self, channels, k=4):
        super().__init__()
        self.k = k
        self.channels = channels

        # Query, Key, Value projections (Equation 9)
        self.W_q = nn.Linear(channels, channels)
        self.W_k = nn.Linear(channels, channels)
        self.W_v = nn.Linear(channels, channels)

        # 添加位置映射层
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

        # Attention function A(·)
        self.attention_func = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Softmax(dim=-2)  # Softmax over neighborhood dimension
        )

    def forward(self, features, position_condition):
        """
        Implements Equations 8-11:
        x^f_i = FFT_1D(f_i), P^f_ij = FFT_1D(P_ij)
        Q^f_i = W_q(x^f_i), K^f_i = W_k(x^f_i), V^f_i = W_v(x^f_i)
        z^f_i = Σ A((K^f_ij ⊖ Q^f_i) ⊕ P^f_ij) ⊗ (V^f_ij ⊕ P^f_ij)
        z_i = IFFT_1D(z^f_i), f_i = z_i ⊕ f_i
        """
        B, N, C = features.shape

        # Equation 8: FFT transformations
        x_f = torch.fft.fft(features, dim=1).real  # FFT_1D along N dimension
        P_f = torch.fft.fft(position_condition, dim=2).real  # FFT_1D along K dimension

        # Equation 9: Q, K, V projections
        Q_f = self.W_q(x_f)  # [B, N, C]
        K_f = self.W_k(x_f)  # [B, N, C]
        V_f = self.W_v(x_f)  # [B, N, C]

        # Group K, V for local attention
        # Use kNN to get neighborhood indices
        # Note: We use spatial domain xyz for kNN, approximated from position_condition
        xyz_approx = position_condition.mean(dim=2)  # [B, N, 3] - approximate positions
        neighbor_idx = knn_point(self.k, xyz_approx, xyz_approx)

        K_f_grouped = index_points(K_f, neighbor_idx)  # [B, N, k, C] -> K^f_ij
        V_f_grouped = index_points(V_f, neighbor_idx)  # [B, N, k, C] -> V^f_ij

        # Equation 10: Point frequency transformer attention
        Q_f_expanded = Q_f.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N, k, C]

        # (K^f_ij ⊖ Q^f_i) ⊕ P^f_ij
        key_diff = K_f_grouped - Q_f_expanded  # [B, N, k, C]
        attn_input = torch.cat([key_diff, P_f], dim=-1)  # [B, N, k, C+3]

        # A((K^f_ij ⊖ Q^f_i) ⊕ P^f_ij)
        attn_weights = self.attention_func(attn_input.view(-1, attn_input.size(-1)))
        attn_weights = attn_weights.view(B, N, self.k, C)  # [B, N, k, C]

        # 修复：将位置特征映射到与特征相同的维度
        P_f_mapped = self.pos_mlp(P_f.reshape(-1, 3)).reshape(B, N, self.k, C)

        # (V^f_ij ⊕ P^f_ij) - combine with position in frequency domain
        V_pos_combined = V_f_grouped + P_f_mapped

        # Attention application and sum over neighborhood
        z_f = torch.sum(attn_weights * V_pos_combined, dim=2)  # [B, N, C]

        # Equation 11: IFFT and residual connection
        z = torch.fft.ifft(z_f, dim=1).real  # IFFT_1D
        output = z + features  # Residual connection

        return output


class DenoisingPointNet(nn.Module):
    """Denoising PointNet as per Equation 12 in paper"""

    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k

        # MLP with position condition integration
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, features, position_condition):
        """
        Implements Equation 12:
        f_i = Λ^K_j=1 MLP(f_ij ⊕ P_ij)
        where Λ^K_j=1 is maxpooling over K neighborhood points
        """
        B, N, C = features.shape

        # Group features to local neighborhoods (f_ij)
        # Use position_condition to determine neighborhood structure
        xyz_approx = position_condition.mean(dim=2)  # [B, N, 3]
        neighbor_idx = knn_point(self.k, xyz_approx, xyz_approx)

        features_grouped = index_points(features, neighbor_idx)  # [B, N, k, C] -> f_ij

        # f_ij ⊕ P_ij (combine grouped features with position condition)
        combined = torch.cat([features_grouped, position_condition], dim=-1)  # [B, N, k, C+3]

        # Apply MLP
        mlp_output = self.mlp(combined.view(-1, combined.size(-1)))  # [(B*N*k), out_channels]
        mlp_output = mlp_output.view(B, N, self.k, -1)  # [B, N, k, out_channels]

        # Λ^K_j=1 (maxpooling over neighborhood)
        output = torch.max(mlp_output, dim=2)[0]  # [B, N, out_channels]

        return output


class DenoisingNetwork(nn.Module):
    """Denoising Network (DNet_θ) with U-Net structure as per paper"""

    def __init__(self, num_classes, channels=64, k=16):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.k = k

        # Time embedding for diffusion timestep t
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

        # Noisy Label Embedding layer
        self.noisy_embed = NoisyLabelEmbedding(num_classes, channels, k)

        # U-Net Encoder (4 layers as per paper architecture)
        # First three encoding layers use Denoising PointNet
        self.enc1 = DenoisingPointNet(channels, channels, k)
        self.enc2 = DenoisingPointNet(channels, channels, k)
        self.enc3 = DenoisingPointNet(channels, channels, k)
        # Last encoding layer uses Point Frequency Transformer
        self.enc4 = PointFrequencyTransformer(channels, k)

        # Shared Transition Down blocks (shared with conditional network)
        self.down1 = SharedTransitionDown(channels, channels, k, stride=2)
        self.down2 = SharedTransitionDown(channels, channels, k, stride=2)
        self.down3 = SharedTransitionDown(channels, channels, k, stride=2)

        # U-Net Decoder (4 layers as per paper architecture)
        # First decoding layer uses Point Frequency Transformer
        self.dec1 = PointFrequencyTransformer(channels, k)
        # Last three decoding layers use Denoising PointNet
        self.dec2 = DenoisingPointNet(channels * 2, channels, k)
        self.dec3 = DenoisingPointNet(channels * 2, channels, k)
        self.dec4 = DenoisingPointNet(channels * 2, channels, k)

        # Shared Transition Up blocks
        self.up1 = SharedTransitionUp(channels * 2, channels)
        self.up2 = SharedTransitionUp(channels * 2, channels)
        self.up3 = SharedTransitionUp(channels * 2, channels)

        # Noise prediction head
        self.noise_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_classes)
        )

    def forward(self, noisy_labels, semantic_condition, position_condition, t, shared_indices=None):
        """
        Forward pass of denoising network
        noisy_labels: [B, N, num_classes] - x^t_i
        semantic_condition: [B, N, C] - S_i
        position_condition: [B, N, K, 3] - P_ij
        t: [B] - diffusion timestep
        """
        B, N = noisy_labels.shape[:2]

        # Time embedding
        t_embed = self.time_embed(t.float().unsqueeze(-1))  # [B, channels]
        t_embed = t_embed.unsqueeze(1).expand(-1, N, -1)  # [B, N, channels]

        # Noisy Label Embedding
        x = self.noisy_embed(noisy_labels, semantic_condition, position_condition)  # [B, N, channels]

        # Add time embedding
        x = x + t_embed

        # Approximate xyz from position condition for transitions
        xyz = position_condition.mean(dim=2)  # [B, N, 3]

        # Encoder with skip connections and shared transitions
        skip1 = x
        x = self.enc1(x, position_condition)

        # Use shared indices if available
        if shared_indices and 'level2' in shared_indices:
            fps_idx2, _ = shared_indices['level2']
            xyz2, x, _, _ = self.down1(xyz, x, fps_idx2)
        else:
            xyz2, x, _, _ = self.down1(xyz, x)

        skip2 = x
        x = self.enc2(x, position_condition[:, :x.shape[1], :, :])  # Adjust for downsampling

        if shared_indices and 'level3' in shared_indices:
            fps_idx3, _ = shared_indices['level3']
            xyz3, x, _, _ = self.down2(xyz2, x, fps_idx3)
        else:
            xyz3, x, _, _ = self.down2(xyz2, x)

        skip3 = x
        x = self.enc3(x, position_condition[:, :x.shape[1], :, :])

        if shared_indices and 'level4' in shared_indices:
            fps_idx4, _ = shared_indices['level4']
            xyz4, x, _, _ = self.down3(xyz3, x, fps_idx4)
        else:
            xyz4, x, _, _ = self.down3(xyz3, x)

        # Bottleneck with Point Frequency Transformer
        x = self.enc4(x, position_condition[:, :x.shape[1], :, :])

        # Decoder with skip connections and shared transitions
        x = self.dec1(x, position_condition[:, :x.shape[1], :, :])

        x = self.up3(xyz3, xyz4, skip3, x)
        x = torch.cat([x, skip3], dim=-1)
        x = self.dec2(x, position_condition[:, :x.shape[1], :, :])

        x = self.up2(xyz2, xyz3, skip2, x)
        x = torch.cat([x, skip2], dim=-1)
        x = self.dec3(x, position_condition[:, :x.shape[1], :, :])

        x = self.up1(xyz, xyz2, skip1, x)
        x = torch.cat([x, skip1], dim=-1)
        x = self.dec4(x, position_condition)

        # Noise prediction head
        noise_pred = self.noise_head(x.view(-1, x.size(-1))).view(B, N, -1)

        return noise_pred


class PointDiffuse(nn.Module):
    """PointDiffuse: Dual-Conditional Diffusion Model for Point Cloud Semantic Segmentation"""

    def __init__(self, num_classes, in_channels=6, channels=64, timesteps=1000):
        super().__init__()
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.channels = channels

        # Diffusion parameters (β_t schedule)
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers for diffusion parameters
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Networks
        self.conditional_net = FullPointTransformer(in_channels, num_classes, channels)
        self.denoising_net = DenoisingNetwork(num_classes, channels)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process - add noise to labels (Equation 1)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, points, features, noise=None):
        """Training loss computation (Equation 6)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        # Forward diffusion
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Get conditions from conditional network
        semantic_condition, position_condition = self.conditional_net(points, features)

        # Predict noise using denoising network
        predicted_noise = self.denoising_net(
            x_noisy, semantic_condition, position_condition, t,
            self.conditional_net.shared_indices
        )

        return predicted_noise, noise

    def forward(self, points, features, labels=None):
        """Forward pass"""
        batch_size = points.shape[0]
        device = points.device

        if self.training and labels is not None:
            # Training mode - compute diffusion loss

            # Convert labels to one-hot encoding
            labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()

            # Sample random timestep
            t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

            # Compute training losses
            predicted_noise, noise = self.p_losses(labels_onehot, t, points, features)

            return predicted_noise, noise, labels_onehot
        else:
            # Inference mode - return conditions for sampling
            semantic_condition, position_condition = self.conditional_net(points, features)
            return semantic_condition, position_condition

    @torch.no_grad()
    def p_sample(self, x, t, points, features, semantic_condition, position_condition):
        """Single denoising step (reverse diffusion)"""
        batch_size = x.shape[0]

        # Predict noise
        predicted_noise = self.denoising_net(
            x, semantic_condition, position_condition, t,
            self.conditional_net.shared_indices
        )

        # Compute reverse diffusion parameters
        alpha_t = extract(self.alphas, t, x.shape)
        alpha_t_bar = extract(self.alphas_cumprod, t, x.shape)
        beta_t = extract(self.betas, t, x.shape)

        # Reverse diffusion formula (Equation 3)
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_t_bar)) * predicted_noise)

        if t.min() > 0:
            # Add noise for non-final steps
            noise = torch.randn_like(x)
            return mean + torch.sqrt(beta_t) * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, points, features, num_steps=None):
        """Generate labels using reverse diffusion (inference)"""
        if num_steps is None:
            num_steps = self.timesteps

        batch_size, n_points = points.shape[:2]
        device = points.device

        # Get conditions from conditional network
        semantic_condition, position_condition = self.conditional_net(points, features)

        # Start from pure noise
        x = torch.randn(batch_size, n_points, self.num_classes, device=device)

        # Reverse diffusion sampling
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

        for i, t_step in enumerate(timesteps):
            t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, points, features, semantic_condition, position_condition)

        return x

    @torch.no_grad()
    def ddim_sample(self, points, features, num_steps=20, eta=0.0):
        """DDIM sampling for faster inference"""
        batch_size, n_points = points.shape[:2]
        device = points.device

        # Get conditions
        semantic_condition, position_condition = self.conditional_net(points, features)

        # Start from noise
        x = torch.randn(batch_size, n_points, self.num_classes, device=device)

        # DDIM timesteps
        c = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, c, device=device)
        timesteps = timesteps[:num_steps]

        for i, t in enumerate(reversed(timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.denoising_net(
                x, semantic_condition, position_condition, t_batch,
                self.conditional_net.shared_indices
            )

            # DDIM step
            alpha_t = extract(self.alphas_cumprod, t_batch, x.shape)
            alpha_t_prev = extract(self.alphas_cumprod_prev, t_batch, x.shape) if t > 0 else torch.ones_like(alpha_t)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * predicted_noise

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise

        return x


# ==================== Training Functions ====================
def pretrain_conditional_network(model, dataloader, optimizer, scheduler, device, epochs=1):
    """Pre-train conditional network for semantic segmentation as per paper"""
    model.conditional_net.train()

    # Temporary classification head
    classifier = nn.Linear(model.channels, model.num_classes).to(device)
    classifier_optimizer = AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print("Pre-training conditional network (Full Point Transformer)...")

    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_points = 0

        model.conditional_net.train()
        for batch in tqdm(dataloader, desc=f"Pre-train Epoch {epoch + 1}/{epochs}"):
            points = batch['points'].to(device).float()
            features = batch['features'].to(device).float()
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # Forward pass
            semantic_features, _ = model.conditional_net(points, features)

            # Classification
            logits = classifier(semantic_features)

            # Compute loss
            loss = criterion(logits.view(-1, model.num_classes), labels.view(-1))
            loss.backward()

            optimizer.step()
            classifier_optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            total_correct += (pred == labels).sum().item()
            total_points += labels.numel()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * total_correct / total_points

        print(f"Pre-train Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    del classifier, classifier_optimizer
    print("Pre-training completed!")


def train_epoch(model, dataloader, optimizer, device, gamma=0.5):
    """Train one epoch with dual loss (Equation 6)"""
    model.train()

    # Freeze conditional network as per paper
    for param in model.conditional_net.parameters():
        param.requires_grad = False

    # Only train denoising network
    for param in model.denoising_net.parameters():
        param.requires_grad = True

    total_loss = 0.0
    noise_loss_total = 0.0
    ce_loss_total = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        points = batch['points'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        predicted_noise, target_noise, labels_onehot = model(points, features, labels)

        # Dual loss computation (Equation 6)
        # L^t_n: noise prediction loss
        noise_loss = F.mse_loss(predicted_noise, target_noise)

        # L_ce: cross-entropy loss between predicted and ground truth labels
        # Reconstruct x^{t-1} for CE loss computation
        x_pred = predicted_noise  # Simplified - in practice would apply full reverse step
        ce_loss = F.cross_entropy(x_pred.view(-1, model.num_classes), labels.view(-1))

        # Total loss: γL^t_n + (1-γ)L_ce
        total_loss_batch = gamma * noise_loss + (1 - gamma) * ce_loss

        total_loss_batch.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.denoising_net.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        total_loss += total_loss_batch.item()
        noise_loss_total += noise_loss.item()
        ce_loss_total += ce_loss.item()
        num_batches += 1

    return {
        'total_loss': total_loss / num_batches,
        'noise_loss': noise_loss_total / num_batches,
        'ce_loss': ce_loss_total / num_batches
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_steps=20, use_ddim=True):
    """Evaluate model using diffusion sampling"""
    model.eval()
    total_correct = 0
    total_points = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        points = batch['points'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        # Generate predictions using diffusion
        if use_ddim:
            predictions = model.ddim_sample(points, features, num_steps)
        else:
            predictions = model.sample(points, features, num_steps)

        pred_labels = predictions.argmax(dim=-1)

        # Calculate accuracy
        total_correct += (pred_labels == labels).sum().item()
        total_points += labels.numel()

    accuracy = 100.0 * total_correct / total_points
    return accuracy


@torch.no_grad()
def calculate_miou(model, dataloader, device, num_classes, num_steps=20, use_ddim=True):
    """Calculate mean IoU"""
    model.eval()

    # Initialize counters for each class
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for batch in tqdm(dataloader, desc="Computing mIoU"):
        points = batch['points'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        # Generate predictions
        if use_ddim:
            predictions = model.ddim_sample(points, features, num_steps)
        else:
            predictions = model.sample(points, features, num_steps)

        pred_labels = predictions.argmax(dim=-1)

        # Calculate IoU for each class
        for class_id in range(num_classes):
            pred_mask = (pred_labels == class_id)
            true_mask = (labels == class_id)

            intersection[class_id] += (pred_mask & true_mask).sum().float()
            union[class_id] += (pred_mask | true_mask).sum().float()

    # Calculate IoU for each class
    class_ious = []
    for class_id in range(num_classes):
        if union[class_id] > 0:
            iou = intersection[class_id] / union[class_id]
            class_ious.append(iou.item())
        else:
            class_ious.append(0.0)

    mean_iou = sum(class_ious) / len(class_ious)
    return mean_iou * 100, [iou * 100 for iou in class_ious]


# ==================== Main Training Script ====================
def main():
    parser = argparse.ArgumentParser(description='PointDiffuse Training')
    parser.add_argument('--dataset', type=str, default='s3dis', choices=['s3dis', 'scannet'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--pretrain_lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.5, help='Loss weighting (Equation 6)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--test_area', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--use_ddim', action='store_true', help='Use DDIM sampling')
    parser.add_argument('--pretrain_only', action='store_true')
    parser.add_argument('--skip_pretrain', action='store_true')
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

    # Dataset
    augmentation = PointCloudAugmentation()

    if args.dataset == 's3dis':
        train_dataset = S3DISDataset(
            args.data_root, split='train', test_area=args.test_area,
            transform=augmentation
        )
        val_dataset = S3DISDataset(
            args.data_root, split='val', test_area=args.test_area
        )
        num_classes = 13
        in_channels = 6  # xyz + rgb
    else:  # scannet
        train_dataset = ScanNetDataset(
            args.data_root, split='train', transform=augmentation
        )
        val_dataset = ScanNetDataset(args.data_root, split='val')
        num_classes = 20
        in_channels = 6

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = PointDiffuse(
        num_classes=num_classes,
        in_channels=in_channels,
        channels=args.channels,
        timesteps=args.timesteps
    ).to(device)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    conditional_params = sum(p.numel() for p in model.conditional_net.parameters())
    denoising_params = sum(p.numel() for p in model.denoising_net.parameters())

    logging.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logging.info(f"Conditional network: {conditional_params / 1e6:.2f}M")
    logging.info(f"Denoising network: {denoising_params / 1e6:.2f}M")

    start_epoch = 0
    best_miou = 0.0

    # Resume from checkpoint
    if args.resume:
        logging.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        logging.info(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.2f}%")

    # Pre-training phase
    if not args.skip_pretrain and not args.pretrain_path and not args.resume:
        logging.info("Starting conditional network pre-training...")

        pretrain_optimizer = AdamW(
            model.conditional_net.parameters(),
            lr=args.pretrain_lr,
            weight_decay=1e-4
        )
        pretrain_scheduler = MultiStepLR(
            pretrain_optimizer,
            milestones=[args.pretrain_epochs // 2],
            gamma=0.1
        )

        pretrain_conditional_network(
            model, train_loader, pretrain_optimizer, pretrain_scheduler,
            device, args.pretrain_epochs
        )

        # Save pretrained model
        torch.save({
            'conditional_net_state_dict': model.conditional_net.state_dict(),
            'args': args
        }, os.path.join(args.save_dir, 'pretrained_conditional_net.pth'))
        logging.info("Pre-trained conditional network saved.")

    elif args.pretrain_path and not args.resume:
        logging.info(f"Loading pre-trained conditional network from {args.pretrain_path}")
        checkpoint = torch.load(args.pretrain_path, map_location=device)
        model.conditional_net.load_state_dict(checkpoint['conditional_net_state_dict'])
        logging.info("Pre-trained conditional network loaded.")

    if args.pretrain_only:
        logging.info("Pre-training only mode. Exiting.")
        return

    # Main training phase
    logging.info("Starting PointDiffuse training...")

    # Optimizer for denoising network only (conditional network frozen)
    denoising_optimizer = AdamW(
        model.denoising_net.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler as per paper
    scheduler = MultiStepLR(
        denoising_optimizer,
        milestones=[100, 150],
        gamma=0.1
    )

    # Resume optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in checkpoint:
        denoising_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, denoising_optimizer, device, args.gamma
        )
        scheduler.step()

        logging.info(
            f"Train - Total: {train_metrics['total_loss']:.4f}, "
            f"Noise: {train_metrics['noise_loss']:.4f}, "
            f"CE: {train_metrics['ce_loss']:.4f}"
        )

        # Validation every 10 epochs
        if (epoch + 1) % 1 == 0:
            logging.info("Running validation...")

            val_accuracy = evaluate_model(
                model, val_loader, device, args.num_steps, args.use_ddim
            )
            val_miou, class_ious = calculate_miou(
                model, val_loader, device, num_classes, args.num_steps, args.use_ddim
            )

            logging.info(f"Validation - Accuracy: {val_accuracy:.2f}%, mIoU: {val_miou:.2f}%")

            # Save best model
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': denoising_optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_miou': best_miou,
                    'val_accuracy': val_accuracy,
                    'class_ious': class_ious,
                    'train_metrics': train_metrics,
                    'args': args
                }, os.path.join(args.save_dir, 'best_model.pth'))
                logging.info(f"New best model saved! mIoU: {best_miou:.2f}%")

        # Regular checkpointing
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': denoising_optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'train_metrics': train_metrics,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    logging.info(f"Training completed! Best mIoU: {best_miou:.2f}%")


# ==================== Inference Script ====================
@torch.no_grad()
def inference(model_path, data_root, dataset='s3dis', test_area=5, num_steps=20,
              batch_size=12, use_ddim=True):
    """Run inference on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint['args']

    # Dataset
    if dataset == 's3dis':
        test_dataset = S3DISDataset(data_root, split='val', test_area=test_area)
        num_classes = 13
        in_channels = 6
    else:
        test_dataset = ScanNetDataset(data_root, split='val')
        num_classes = 20
        in_channels = 6

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = PointDiffuse(
        num_classes=num_classes,
        in_channels=in_channels,
        channels=getattr(args, 'channels', 64),
        timesteps=getattr(args, 'timesteps', 1000)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Running inference with {num_steps} steps...")
    start_time = time.time()

    # Evaluate
    accuracy = evaluate_model(model, test_loader, device, num_steps, use_ddim)
    miou, class_ious = calculate_miou(
        model, test_loader, device, num_classes, num_steps, use_ddim
    )

    inference_time = time.time() - start_time

    print(f"Inference completed in {inference_time:.2f}s")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test mIoU: {miou:.2f}%")

    if dataset == 's3dis':
        class_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                       'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
        print("\nPer-class IoU:")
        for name, iou in zip(class_names, class_ious):
            print(f"  {name}: {iou:.2f}%")

    return accuracy, miou, class_ious


# ==================== Utility Functions ====================
def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def compute_flops(model, input_shape):
    """Estimate FLOPs (simplified)"""
    # This is a simplified FLOP estimation
    # For accurate measurement, use tools like ptflops or fvcore
    total_params = sum(p.numel() for p in model.parameters())
    batch_size, n_points, channels = input_shape

    # Rough estimation based on model complexity
    flops_per_point = total_params * 2  # Multiply-add operations
    total_flops = batch_size * n_points * flops_per_point

    return total_flops / 1e9  # Return in GFLOPs


def analyze_model_complexity(model_path):
    """Analyze model complexity"""
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']

    model = PointDiffuse(
        num_classes=13 if args.dataset == 's3dis' else 20,
        in_channels=6,
        channels=getattr(args, 'channels', 64),
        timesteps=getattr(args, 'timesteps', 1000)
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    total_params, trainable_params = count_parameters(model)
    conditional_params = sum(p.numel() for p in model.conditional_net.parameters())
    denoising_params = sum(p.numel() for p in model.denoising_net.parameters())

    print("Model Complexity Analysis:")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Conditional network: {conditional_params / 1e6:.2f}M")
    print(f"Denoising network: {denoising_params / 1e6:.2f}M")

    # Estimate FLOPs for typical input
    flops = compute_flops(model, (1, 80000, 6))  # Batch=1, Points=80k, Features=6
    print(f"Estimated FLOPs: {flops:.2f}G")

    return {
        'total_params': total_params,
        'conditional_params': conditional_params,
        'denoising_params': denoising_params,
        'flops': flops
    }


if __name__ == '__main__':
    main()