import torch
import torch.nn as nn
from adjacency import get_vsl_adjacency

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1, residual=True):
        super(STGCN_Block, self).__init__()
        
        # Spatial Component (Learnable Edge Importance)
        self.register_buffer('A', torch.from_numpy(adjacency).float())
        self.edge_importance = nn.Parameter(torch.ones(adjacency.shape))
        
        self.scn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Temporal Component (9-frame kernel)
        self.tcn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=(9, 1), 
                      stride=(stride, 1), 
                      padding=(4, 0)),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual Connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2) # Increased slightly for 400 classes

    def forward(self, x):
        # x shape: (N, C, T, V)
        res = self.residual(x)
        
        # 1. Spatial Phase (Multiply Adjacency by Learnable Weights)
        # Using einsum for fast parallel graph convolution
        x = torch.einsum('nctv,vw->nctw', x, self.A * self.edge_importance)
        x = self.scn(x)
        
        # 2. Temporal Phase
        x = self.tcn(x)
        
        # 3. Fuse & Activate
        x = self.relu(x + res)
        return self.dropout(x)
    
class LateFusion_STGCN(nn.Module):
    def __init__(self, num_classes=400):
        super().__init__()
        A = get_vsl_adjacency(42)

        self.joint_bn = nn.BatchNorm1d(3 * 42)
        self.bone_bn = nn.BatchNorm1d(3 * 42)
        self.motion_bn = nn.BatchNorm1d(3 * 42)
        
        # Branch 1: Joint Stream (extracts position)
        self.joint_stream = nn.ModuleList([
            STGCN_Block(3, 64, A, residual=False),
            STGCN_Block(64, 128, A, stride=2),
            STGCN_Block(128, 256, A, stride=2)
        ])
        
        # Branch 2: Bone Stream (extracts relative movement)
        self.bone_stream = nn.ModuleList([
            STGCN_Block(3, 64, A, residual=False),
            STGCN_Block(64, 128, A, stride=2),
            STGCN_Block(128, 256, A, stride=2)
        ])

        self.motion_stream = nn.ModuleList([
            STGCN_Block(3, 64, A, residual=False),
            STGCN_Block(64, 128, A, stride=2),
            STGCN_Block(128, 256, A, stride=2)
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # The Fusion Layer: Combines 256 (Joint) + 256 (Bone) + 256 (Motion) = 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
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
        # Process Joint Stream
# 2. Process Streams
        for j_block, b_block, m_block in zip(self.joint_stream, self.bone_stream, self.motion_stream):
            x_joint = j_block(x_joint)
            x_bone = b_block(x_bone)
            x_motion = m_block(x_motion)

        # 3. Global Pooling and Fusion
        feat_j = self.pool(x_joint).view(x_joint.size(0), -1)
        feat_b = self.pool(x_bone).view(x_bone.size(0), -1)
        feat_m = self.pool(x_motion).view(x_motion.size(0), -1)
        
        return self.classifier(torch.cat([feat_j, feat_b, feat_m], dim=1))