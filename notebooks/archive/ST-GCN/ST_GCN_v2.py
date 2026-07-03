import torch
import torch.nn as nn
from adjacency import get_vsl_adjacency


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) # Amplifies important channels

# -------------------------
# UPGRADED: STGCN Block
# -------------------------
class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1, residual=True):
        super(STGCN_Block, self).__init__()
        
        # Spatial Component (Physical + Mask + Fully Learnable)
        self.register_buffer('A', adjacency.float())
        self.edge_importance = nn.Parameter(torch.ones(adjacency.shape))
        self.B = nn.Parameter(torch.zeros(adjacency.shape)) # NEW: Fully adaptive connections
        
        self.scn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 3, kernel_size=1),
            nn.BatchNorm2d(out_channels * 3)
        )
        
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=(9, 1), 
                      stride=(stride, 1), 
                      padding=(4, 0)),
            nn.BatchNorm2d(out_channels)
        )
        
        self.se = SELayer(out_channels) # NEW: Attention Mechanism
        
        self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                    nn.BatchNorm2d(out_channels)
                ) if not (in_channels == out_channels and stride == 1) else nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3) 

    def forward(self, x):
        res = self.residual(x)

        x = self.scn(x)

        N, KC, T, V = x.size()
    
        # Expand features for 3 spatial channels
        # Assuming self.scn now outputs out_channels * 3
        x_spatial = x.view(N, 3, KC // 3, T, V)
        
        # Phase 1: Spatial Graph Conv (A * mask + B)
        adaptive_graph = (self.A * self.edge_importance) + self.B
        x = torch.einsum('nactv,avw->nactw', x_spatial, adaptive_graph)
        x = x.sum(dim=1) # Sum over the 3 spatial channels
        
        # Phase 2: Temporal Conv
        x = self.tcn(x)
        
        # Phase 3: Attention & Fuse
        x = self.se(x) # Apply attention before adding residual
        x = self.relu(x + res)
        
        return self.dropout(x)


class LateFusion_STGCN(nn.Module):
    def __init__(self, num_classes=400):
        super().__init__()
        A = get_vsl_adjacency(42)

        self.joint_bn = nn.BatchNorm1d(3 * 42)
        self.bone_bn = nn.BatchNorm1d(3 * 42)
        self.motion_bn = nn.BatchNorm1d(3 * 42)
        
        # Helper function to generate a 9-block deep stream
        def make_stream():
            return nn.ModuleList([
                STGCN_Block(3, 64, A, residual=False),
                STGCN_Block(64, 64, A),
                STGCN_Block(64, 64, A),
                STGCN_Block(64, 128, A, stride=2),
                STGCN_Block(128, 128, A),
                STGCN_Block(128, 128, A),
                STGCN_Block(128, 256, A, stride=2),
                STGCN_Block(256, 256, A),
                STGCN_Block(256, 256, A)
            ])

        self.joint_stream = make_stream()
        self.bone_stream = make_stream()
        self.motion_stream = make_stream()

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased dropout slightly for the deeper network
            nn.Linear(512, num_classes)
        )

    def _normalize(self, x, bn):
        N, C, T, V = x.size()
        x = x.permute(0, 1, 3, 2).reshape(N, C*V, T)
        x = bn(x)
        return x.view(N, C, V, T).permute(0, 1, 3, 2)

    def forward(self, x_joint, x_bone, x_motion):
        x_joint = self._normalize(x_joint, self.joint_bn)
        x_bone = self._normalize(x_bone, self.bone_bn)
        x_motion = self._normalize(x_motion, self.motion_bn)

        # Process Streams
        for j_block, b_block, m_block in zip(self.joint_stream, self.bone_stream, self.motion_stream):
            x_joint = j_block(x_joint)
            x_bone = b_block(x_bone)
            x_motion = m_block(x_motion)

        # Global Pooling and Fusion
        feat_j = self.pool(x_joint).view(x_joint.size(0), -1)
        feat_b = self.pool(x_bone).view(x_bone.size(0), -1)
        feat_m = self.pool(x_motion).view(x_motion.size(0), -1)
        
        return self.classifier(torch.cat([feat_j, feat_b, feat_m], dim=1))