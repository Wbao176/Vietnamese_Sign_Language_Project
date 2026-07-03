import torch
import torch.nn as nn

class SAConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, num_classes=10):
        super(SAConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        
        # 1. ConvLSTM Gate Logic (The "Conv" part)
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=self.padding)
        
        # 2. Self-Attention Logic (The "SA" part from your study)
        self.query = nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1)
        self.key   = nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim,      kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable wf parameter
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_dropout = nn.Dropout2d(p=0.3)
        self.feat_dropout = nn.Dropout(p=0.5)
        # 3. Final Classifier (The "Logic" for prediction)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def apply_attention(self, h_t):
        batch, c, h, w = h_t.size()
        
        # Projection (Eq 6)
        proj_query = self.query(h_t).view(batch, -1, h * w).permute(0, 2, 1)
        proj_key   = self.key(h_t).view(batch, -1, h * w)
        
        # Similarity and Normalization (Eq 7)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        
        # Aggregation and Residual Connection (Eq 8 & wf addition)
        proj_value = self.value(h_t).view(batch, -1, h * w) 
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        
        return self.gamma * out + h_t

    def forward(self, x):
        # x shape: (Batch, 60, 1, 8, 8)
        batch_size, seq_len, _, h, w = x.size()
        
        # Initial states
        h_t = torch.zeros(batch_size, self.hidden_dim, h, w).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, h, w).to(x.device)

        # Iterate through the 60-frame sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            combined = torch.cat([x_t, h_t], dim=1)
            combined = self.spatial_dropout(combined)  # Apply spatial dropout to the combined input
            # Standard LSTM Gates
            gates = self.conv(combined)
            i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)
            
            c_t = torch.sigmoid(f) * c_t + torch.sigmoid(i) * torch.tanh(g)
            h_t = torch.sigmoid(o) * torch.tanh(c_t)
            
            # Apply Self-Attention to the hidden state at each step
            h_t = self.apply_attention(h_t)
            h_t = self.spatial_dropout(h_t)
        
        # Global spatial information flattened for the final output
        out = self.global_pool(h_t)    # Shape: (Batch, 64, 1, 1)
        out = out.view(batch_size, -1) # Shape: (Batch, 64)        
        
        out = self.feat_dropout(out)
        out = self.fc(out)   
        return out


class MultiViewSAConvlLSTM(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=400):
        super(MultiViewSAConvlLSTM, self).__init__()
        self.front_model = SAConvLSTM(input_dim=1, hidden_dim=hidden_dim, num_classes=num_classes)
        self.left_model  = SAConvLSTM(input_dim=1, hidden_dim=hidden_dim, num_classes=num_classes)
        self.right_model = SAConvLSTM(input_dim=1, hidden_dim=hidden_dim, num_classes=num_classes)
        
        # Final fusion layer to combine the three views
        fused_dim = 64 * 3  # 3 views concatenated

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_front, x_left, x_right):
        # x shapes: (Batch, 60, 1, 8, 8)
        
        # Get features from each view's SA-ConvLSTM
        # Note: Modify your SAConvLSTM to return 'out' before the FC layer if preferred
        feat_f = self.extract_features(self.front_model, x_front)
        feat_l = self.extract_features(self.left_model, x_left)
        feat_r = self.extract_features(self.right_model, x_right)

        # Late Fusion: Concatenate the three views
        fused = torch.cat([feat_f, feat_l, feat_r], dim=1) # (Batch, fused_dim)

        return self.classifier(fused)
    
    def extract_features(self, model, x):
        # Temporary helper to get the flattened state before the model's own FC
        # You can also modify your class to have a 'return_features' flag
        batch_size, seq_len, _, h, w = x.size()
        h_t = torch.zeros(batch_size, model.hidden_dim, h, w).to(x.device)
        c_t = torch.zeros(batch_size, model.hidden_dim, h, w).to(x.device)

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            combined = torch.cat([x_t, h_t], dim=1)
            gates = model.conv(combined)
            i, f, g, o = torch.split(gates, model.hidden_dim, dim=1)
            c_t = torch.sigmoid(f) * c_t + torch.sigmoid(i) * torch.tanh(g)
            h_t = torch.sigmoid(o) * torch.tanh(c_t)
            h_t = model.apply_attention(h_t)
            h_t = model.spatial_dropout(h_t)
        
        feat = model.global_pool(h_t)
            
        return feat.view(x.size(0), -1)