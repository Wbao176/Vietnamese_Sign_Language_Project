import torch
from torch import nn

from notebooks.archive.Model_Essemble.SA_CNN_LSTM_model import MultiViewSAConvlLSTM
from notebooks.archive.Model_Essemble.ST_GCN_v2 import LateFusion_STGCN


class VSL_Master_Fusion(nn.Module):
    def __init__(self, gcn_weights_path, sa_weights_path, num_classes=400):
        super().__init__()
        self.gcn_stream = LateFusion_STGCN(num_classes=num_classes)
        self.sa_stream = MultiViewSAConvlLSTM(hidden_dim=64, num_classes=num_classes)

        # Load weights
        self.gcn_stream.load_state_dict(torch.load(gcn_weights_path))
        self.sa_stream.load_state_dict(torch.load(sa_weights_path))

        # Combined classifier: 768 (GCN) + 192 (Multi-View SA) = 960
        self.final_head = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_f, x_l, x_r, x_j, x_b, x_m):
        # 1. Get GCN Features (Need to bypass GCN's own classifier)
        # We manually run the normalize and streams to get the pooled features
        feat_j = self.gcn_stream.pool(self.get_gcn_feat(x_j, self.gcn_stream.joint_stream, self.gcn_stream.joint_bn)).view(x_j.size(0), -1)
        feat_b = self.gcn_stream.pool(self.get_gcn_feat(x_b, self.gcn_stream.bone_stream, self.gcn_stream.bone_bn)).view(x_b.size(0), -1)
        feat_m = self.gcn_stream.pool(self.get_gcn_feat(x_m, self.gcn_stream.motion_stream, self.gcn_stream.motion_bn)).view(x_m.size(0), -1)
        gcn_fused = torch.cat([feat_j, feat_b, feat_m], dim=1) # 768

        # 2. Get Phonetic Features
        sa_f = self.sa_stream.extract_features(self.sa_stream.front_model, x_f)
        sa_l = self.sa_stream.extract_features(self.sa_stream.left_model, x_l)
        sa_r = self.sa_stream.extract_features(self.sa_stream.right_model, x_r)
        sa_fused = torch.cat([sa_f, sa_l, sa_r], dim=1) # 192

        # 3. Final Fusion
        total_fused = torch.cat([gcn_fused, sa_fused], dim=1)
        return self.final_head(total_fused)

    def get_gcn_feat(self, x, stream, bn):
        x = self.gcn_stream._normalize(x, bn)
        for block in stream:
            x = block(x)
        return x