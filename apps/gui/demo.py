import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d

# IMPORTANT: Import your phonetic extractors!
from notebooks.archive.GUI.phonetic_extraction import (
    extract_hand_shape, 
    extract_palm_orientation, 
    extract_hand_location, 
    extract_motion_features_from_sequence
)

# --- 1. MEDIAPIPE LOGIC ---
mp_holistic = mp.solutions.holistic
UPPER_BODY_POSE_LANDMARKS = 25 
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return image, results

def extract_keypoints(results):
    """
    MODIFIED: We flatten the arrays here so they perfectly match your 
    training script's 201-element format. 
    Pose = 0:75, Left Hand = 75:138, Right Hand = 138:201
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:UPPER_BODY_POSE_LANDMARKS]]).flatten() if results.pose_landmarks else np.zeros(UPPER_BODY_POSE_LANDMARKS * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(LEFT_HAND_LANDMARKS * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(RIGHT_HAND_LANDMARKS * 3)
    
    # Return as (201,) 
    return np.concatenate([pose, lh, rh])

def interpolate_sequence(sequence, target_len=60):
    """
    MODIFIED: Interpolates the 1D (201) feature vectors directly.
    """
    if len(sequence) < 2: return None
    sequence = np.array(sequence) # Shape: [Current_Len, 201]
    curr_len, features = sequence.shape
    
    old_x = np.linspace(0, 1, curr_len)
    new_x = np.linspace(0, 1, target_len)
    
    interpolated = np.zeros((target_len, features))
    for f in range(features):
        interp_func = interp1d(old_x, sequence[:, f], kind='cubic')
        interpolated[:, f] = interp_func(new_x)
        
    return interpolated # Shape: [60, 201]

def extract_live_phonetic_features(interpolated_sequence):
    """
    This is your batch processing logic adapted for a single live sequence.
    Takes in (60, 201) -> Outputs a PyTorch Tensor of (1, 60, 64)
    """
    T = len(interpolated_sequence)
    motion = extract_motion_features_from_sequence(interpolated_sequence)
    
    new_seq = []
    
    for i in range(T):
        frame = interpolated_sequence[i]
        left_hand = frame[75:138]  # (63,)
        right_hand = frame[138:201] # (63,)

        # Left Hand Phonetics
        if np.all(left_hand == 0):
            s_l, n_l, l_l = np.zeros(19), np.zeros(3), np.zeros(3)
        else:
            s_l = extract_hand_shape(left_hand)          
            n_l = extract_palm_orientation(left_hand)    
            l_l = extract_hand_location(left_hand)       

        # Right Hand Phonetics
        if np.all(right_hand == 0):
            s_r, n_r, l_r = np.zeros(19), np.zeros(3), np.zeros(3)
        else:
            s_r = extract_hand_shape(right_hand)          
            n_r = extract_palm_orientation(right_hand)    
            l_r = extract_hand_location(right_hand)       

        m = motion[i] 

        # Combine into the 64-dimensional feature vector
        new_frame = np.concatenate([s_l, n_l, l_l, s_r, n_r, l_r, m]) 
        new_seq.append(new_frame)

    # Convert to Tensor and add Batch Dimension for the SA-CNN-LSTM model
    # Shape becomes [1, 60, 64]
    return torch.FloatTensor(np.array(new_seq)).unsqueeze(0)