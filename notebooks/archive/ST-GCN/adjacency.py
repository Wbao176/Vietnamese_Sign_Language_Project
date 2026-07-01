import numpy as np
import torch

def get_vsl_adjacency(num_nodes=42):
    # MediaPipe Hand Connections (0-20)
    # The first node in each tuple is the "parent" (closer to wrist)
    finger_base = [(0,1), (1,2), (2,3), (3,4),  # Thumb
                   (0,5), (5,6), (6,7), (7,8),  # Index
                   (0,9), (9,10), (10,11), (11,12), # Middle
                   (0,13), (13,14), (14,15), (15,16), # Ring
                   (0,17), (17,18), (18,19), (19,20)] # Pinky
    
    # A has shape (3, 42, 42)
    # Channel 0: Self-loops
    # Channel 1: Centripetal (Neighbor -> Parent)
    # Channel 2: Centrifugal (Parent -> Neighbor)
    A = np.zeros((3, num_nodes, num_nodes))
    
    # 1. Fill Self-loops (Channel 0)
    for i in range(num_nodes):
        A[0, i, i] = 1
        
    # 2. Fill Spatial Connections
    for i, j in finger_base:
        # Left Hand
        A[1, j, i] = 1 # j to i is toward wrist
        A[2, i, j] = 1 # i to j is toward fingertip
        
        # Right Hand (+21)
        A[1, j+21, i+21] = 1
        A[2, i+21, j+21] = 1
        
    # 3. Connect Wrists (Channel 0)
    A[0, 0, 21] = A[0, 21, 0] = 1
    
    # 4. Normalization (Row-normalization)
    # We normalize each channel independently
    A_norm = np.zeros_like(A)
    for c in range(3):
        row_sum = np.sum(A[c], axis=1)
        d_inv = np.zeros_like(row_sum)
        # Avoid division by zero
        d_inv[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
        D_inv = np.diag(d_inv)
        A_norm[c] = D_inv @ A[c]
        
    return torch.from_numpy(A_norm).float()