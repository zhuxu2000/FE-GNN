from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import torch
import torch.nn as nn
from torch_geometric.typing import Adj,Tensor, OptTensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F

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

def attention(query,key,value,mask=None):
    d_model=key.shape[-1]
    att_ = torch.matmul(query,key.transpose(-2,-1))/d_model**0.5

    if mask is not None:
        att_ = att_.masked_fill_(mask,-1e9)
    att_score = torch.softmax(att_,-1)
    return torch.matmul(att_score,value)

class MultiHeadAttention(nn.Module):
    def __init__(self,heads,d_model,dropout=0.1):
        super().__init__()
        assert d_model % heads ==0
        #不投影会退化成线性的格式
        self.q_linear = nn.Linear(d_model,d_model,bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model,d_model,bias=False)
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.d_k = d_model//heads
        self.d_model = d_model

    def forward(self,q,k,v):
        #将q，k，v拆成多头
        #n,seq_len,d_model ->[n,head,seq_len,d_k]
        q = self.q_linear(q).reshape(q.shape[0],self.heads,self.d_k).transpose(1,2)#bach维度不变，有多少个头，头的维度
        k = self.k_linear(k).reshape(k.shape[0],self.heads,self.d_k).transpose(1,2)
        v = self.v_linear(v).reshape(v.shape[0],self.heads,self.d_k).transpose(1,2)
        #再将[n,head,seq_len,d_k] -->[n,seq_len,d_model]
        out = attention(q,k,v)
        out = out.transpose(1,2).reshape(out.shape[0],self.d_model)
        out = self.linear(out)
        return out

class enhance_mr_mp(nn.Module):
    def __init__(self, alpha=1.0, reward_thresh=0.2, adaptive=True):
        super().__init__()
        self.alpha = alpha
        self.reward_thresh = reward_thresh
        self.adaptive = adaptive

        if adaptive:
            self.reward_weight = nn.Parameter(torch.tensor(2.0))
            self.punish_weight = nn.Parameter(torch.tensor(1.0))
            self.reward_thresh = nn.Parameter(torch.tensor(reward_thresh))

    def forward(self, x):
        m1 = torch.tanh(x)
        m2 = x

        reward = torch.where(m1 < self.reward_thresh,
                             torch.zeros_like(m1),
                             m1 * self.reward_weight)

        punish = self.alpha * (torch.exp(m2) - 1)
        punish = torch.where(m2 > 0, m2, punish * self.punish_weight)
        out = (reward * punish) + punish
        return out

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

    def forward(self, features: list[Tensor],x_att) -> Tensor:
        x_ce = self.channel_encoder(features)
        x_fused = self.cross_domain_fuse(torch.cat([x_att, x_ce], dim=0))
        return x_fused.squeeze(0)


class Adaptive_Conv(MessagePassing):
    def __init__(self):
        super().__init__(node_dim=0, aggr='add')
        self.weight_controller = MultiHeadAttention(8, 64)
        self.avtive_cove = enhance_mr_mp(alpha=1.0, reward_thresh=0.2)
        self.lin = nn.Linear(64 * 3, 64, bias=False)
        self.lin_edge = nn.Linear(64, 64, bias=False)
        self.lin_no_edge = nn.Linear(64 * 2, 64, bias=False)

    def forward(self, x, edge_index, edge_attr=None):
        # Track if edge attributes are provided
        self.has_edge_attr = edge_attr is not None
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr=None):
        message_nei = self.weight_controller(x_i, x_j, x_j)
        message_self = self.weight_controller(x_i, x_j, x_i)

        if self.has_edge_attr:
            # Concatenate self message, edge attribute, and neighbor message
            vec = torch.cat((message_self, edge_attr, message_nei), 1)
            block_score = self.avtive_cove(vec)
            block_score = self.lin(block_score)
        else:
            # Concatenate only self and neighbor messages when no edge attributes
            vec = torch.cat((message_self, message_nei), 1)
            block_score = self.avtive_cove(vec)
            block_score = self.lin_no_edge(block_score)

        return block_score

class Adaptive_Enhancement_Conv(nn.Module):
    def __init__(self, node_feature, edge_feature, out_channels, num_layers):
        super().__init__()
        self.node_encoder = nn.Linear(node_feature, out_channels)
        # Flag to track if edge features are expected
        self.has_edge_feature = edge_feature > 0
        if self.has_edge_feature:
            self.edge_encoder = nn.Linear(edge_feature, out_channels)
        self.num_layers = num_layers
        self.atom_convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            conv = Adaptive_Conv()
            self.atom_convs.append(conv)
        self.projection = torch.nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        features = []
        x = self.node_encoder(x)
        # Process edge features only if expected and provided
        if self.has_edge_feature and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None

        for i in range(0, self.num_layers):
            h = self.atom_convs[i](x, edge_index, edge_attr)
            features.append(h)

        features = torch.stack(features, dim=1).transpose(0, 1)
        return features

class FE_GNN(nn.Module):
    """FE-GNN: Topology-Aware Molecular Graph Representation Learning with Multi-Scale Feature Enhancement

    Args:
        in_dim (int): Input feature dimension
        hid_dim (int): Hidden layer dimension
        out_dim (int): Output prediction dimension

    Architecture:
        1. Hierarchical Graph Encoder
        2. Dual Attention Module (Channel + Spatial)
        3. Cross-Domain Stabilization (Feature Fusion Unit)
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        torch.manual_seed(12345)

        # Graph Encoding Backbone
        self.encoder = Adaptive_Enhancement_Conv(node_feature=in_dim,edge_feature=6 ,out_channels=64,num_layers=3)

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
        features = self.encoder(x,edge_index, edge_attr)

        # 2. Attention-Guided Feature
        x_ca = self.channel_att(features)
        x_sa = self.spatial_att(features)
        x_att = features * x_ca * x_sa

        # 3. Cross-Domain Stabilization
        x_fused = self.fusion(features,x_att)

        # 4. Global Pooling & Classification
        x_pool = global_mean_pool(x_fused, batch)
        return self.classifier(x_pool)


