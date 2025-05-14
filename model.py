from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import torch
import torch.nn as nn
from torch_geometric.typing import Adj,Tensor, OptTensor


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(3, 64, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))  # Apply average pooling to compress global spatial information: (B, C, H, W) --> (B, C, 1, 1)
        max_out = self.mlp(self.max_pool(x))  # Apply max pooling to compress global spatial information: (B, C, H, W) --> (B, C, 1, 1)
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x):
        avg_out = torch.mean(x, dim=0, keepdim=True)
        max_out, _ = torch.max(x, dim=0, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=0)
        x = self.conv1(x)
        return self.sigmoid(x)

class CategoryAttentionBlock(nn.Module):
    def __init__(self, in_channels, classes, k):
        super(CategoryAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, k * classes, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(k * classes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classes = classes
        self.k = k

    def forward(self, inputs):
        chanel, node, feature = inputs.size()

        # Convolution, Batch Normalization, and ReLU
        F = self.conv(inputs)
        F1 = self.relu(F)

        # Global Max Pooling
        x = self.max_pool(F1)
        x = x.view(self.classes,self.k)
        # Compute the attention vector S
        S = x.mean(dim=-1, keepdims=False)
        S = S.view(-1,1,1)

        # Reshape and mean pooling along the 'k' dimension
        x = F1.view(self.classes,self.k,node,feature)

        x = x.mean(dim=1, keepdims=False)
        # Element-wise multiplication
        x = S * x
        # Mean pooling along the 'k' dimension to get the final attention mask M
        M = x.mean(dim=0, keepdims=True)

        # Final multiplication with the input to get the output
        semantic = inputs * M

        return semantic

class Eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(Eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class FeatureFusionUnit(nn.Module):
    """Multi-scale feature fusion module with cross-domain stabilization"""

    def __init__(self, in_channels: int=3, mid_channels: int = 64):
        super().__init__()
        self.channel_encoder= nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=7, padding=3),
            nn.ReLU()
        )
        self.cross_domain_fuse = nn.Conv2d(2 * in_channels, 1, kernel_size=7, padding=3)

    def reset_parameters(self):
        for layer in self.channel_encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.cross_domain_fuse.reset_parameters()

    def forward(self, features,x_att:Tensor) -> Tensor:
        x_ce = self.channel_encoder(features)
        x_fused = self.cross_domain_fuse(torch.cat([x_att, x_ce], dim=0))
        return x_fused.squeeze(0)

class GraphEncoder(nn.Module):
    """Hierarchical graph representation learning backbone"""

    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            GCNConv(in_dim, hid_dim),
            GCNConv(hid_dim, hid_dim)
        ])
        self.gat = GATConv(hid_dim, hid_dim)

    def reset_parameters(self):
        for layer in self.gcn_layers:
            layer.reset_parameters()
        self.gat.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        features = []
        for conv in self.gcn_layers:
            x = conv(x, edge_index).relu()
            features.append(x)
        x = self.gat(features[-1], edge_index, edge_attr)
        features.append(x)
        features = torch.stack(features, dim=1).transpose(0,1)

        return features

class FE_GNN(nn.Module):
    """FE-GNN: Topology-Aware Molecular Graph Representation Learning with Multi-Scale Feature Enhancement

    Args:
        in_dim (int): Input feature dimension
        hid_dim (int): Hidden layer dimension
        out_dim (int): Output prediction dimension

    Architecture:
        1. Hierarchical Graph Encoder (GCN/GAT)
        2. Dual Attention Module (Channel + Spatial)
        3. Cross-Domain Stabilization (Feature Fusion Unit)
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        torch.manual_seed(12345)

        # Graph Encoding Backbone
        self.encoder = GraphEncoder(in_dim, hid_dim)

        # Attention Mechanisms
        self.channel_att = ChannelAttention()
        self.spatial_att = SpatialAttention()

        # Feature Fusion
        self.fusion = FeatureFusionUnit(in_channels=3)

        # Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, out_dim)
        )

    def reset_parameters(self):
        """Xavier initialization for all learnable parameters"""
        self.encoder.reset_parameters()
        self.channel_att.reset_parameters()
        self.spatial_att.reset_parameters()
        self.fusion.reset_parameters()
        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, batch: OptTensor = None) -> Tensor:
        # 1. Multi-Scale Feature Extraction
        features = self.encoder(x, edge_index, edge_attr)

        # 2. Attention-Guided Feature
        x_ca = self.channel_att(features)
        x_sa = self.spatial_att(features)
        x_att = features * x_ca * x_sa

        # 3. Cross-Domain Stabilization
        x_fused = self.fusion(features,x_att)

        # 4. Global Pooling & Classification
        x_pool = global_mean_pool(x_fused, batch)
        return self.classifier(x_pool)