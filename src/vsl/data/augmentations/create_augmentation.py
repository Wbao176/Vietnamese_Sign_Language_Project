from datetime import datetime
import pandas as pd
import random

import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d
from tqdm import tqdm

import os
import json
import cv2

from notebooks.archive.augmentations.augmentation import scale_sequence_function, rotate_sequence_function, translate_sequence_function, time_stretch_function, inverse_kinematics_function, inter_hand_distance_function


mp_holistic = mp.solutions.holistic
UPPER_BODY_POSE_LANDMARKS = 25   # Maybe change it to 33 (because the upper body (25) in new version is may not working)
LEFT_HAND_LANDMARKS = 21
RIGHT_HAND_LANDMARKS = 21

TOTAL_LANDMARKS = UPPER_BODY_POSE_LANDMARKS + LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS

ALL_POSE_CONNECTION = list(mp_holistic.POSE_CONNECTIONS)  # Take out a set of index pairs & convert them to a list
UPPER_BODY_POSE_CONNECTIONS = [] # Create an empty list only store upper body connections

for connection in ALL_POSE_CONNECTION: # connection = (start_idx, end_idx)
    if connection[0] < UPPER_BODY_POSE_LANDMARKS and connection[1] < UPPER_BODY_POSE_LANDMARKS: # Just tak
        UPPER_BODY_POSE_CONNECTIONS.append(connection)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose_kps = np.zeros((UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((LEFT_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((RIGHT_HAND_LANDMARKS, 3))

    if results and results.pose_landmarks:
        for i in range(UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]
    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])

    keypoints = np.concatenate([pose_kps,left_hand_kps, right_hand_kps])

    return keypoints.flatten()

def interpolate_keypoints(keypoints_sequence, target_len = 60):
    if len(keypoints_sequence) == 0:
        return None

    org_time = np.linspace(0, 1, len(keypoints_sequence)) # Normalize orginal time frame [0 -> 1]
    target_time = np.linspace(0, 1, target_len) # Normalize target time frame [0 -> 1]
    feature_dim = keypoints_sequence[0].shape[0] # Get the dimension of keypoints (in this case: 25 pose + 21 left hand + 21 right hand = 67 keypoints * 3 (x,y,z) = 201)
    interpolated_sequence = np.zeros((target_len, feature_dim)) # Create an empty array to store the interpolated keypoints

    for i in range(feature_dim):
        feature_values = [frame[i] for frame in keypoints_sequence] # Take out the value of the i-th feature across all frames

        interpolator = interp1d(
            org_time, feature_values,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        interpolated_sequence[:, i] = interpolator(target_time) # Interpolate the i-th feature across the target time frame
      
    return interpolated_sequence


def process_video(video_path, holistic):
    sequence_frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, total_frames // 100)  # Calculate the step to sample frames (max 100 frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue  # Skip frames based on the calculated step

        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)

            if keypoints is not None:
                sequence_frames.append(keypoints)

        except Exception as e:
            continue

    cap.release()
    return sequence_frames


def create_folder(data_path, action):
    action_folder = os.path.join(data_path, action)
    os.makedirs(action_folder, exist_ok=True)
    return action_folder

class Gettime():
    def __init__(self):
        self.start_time = datetime.now()
    
    def get_time(self):
        return datetime.now() - self.start_time
    

augmentations = [
    scale_sequence_function,
    rotate_sequence_function,
    translate_sequence_function,
    time_stretch_function,
    inter_hand_distance_function
]


def create_augmentation(original_sequence, augmentation_functions, num_generate_samples: int, max_aug_per_sample: int):
    generated_samples = []
    if not original_sequence or not augmentation_functions:
        return generated_samples

    num_available_augs = len(augmentation_functions)

    for i in range(num_generate_samples):
        current_sequence = [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in original_sequence]  # Deep copy of the original sequence

        num_augs_to_apply = random.randint(1, min(max_aug_per_sample, num_available_augs))

        selected_augs_idx = random.sample(range(num_available_augs), num_augs_to_apply)
        selected_augs = [augmentation_functions[idx] for idx in selected_augs_idx]

        random.shuffle(selected_augs)

        for aug in selected_augs:
            current_sequence = aug(current_sequence)

            if not current_sequence or all(frame is None for frame in current_sequence):
                break # If the augmented sequence is invalid, break out of the loop and skip adding it to the generated samples

        if not current_sequence or all(frame is None for frame in current_sequence):
            continue # If the augmented sequence is valid, skip adding it to the generated samples

        generated_samples.append(current_sequence)

    return generated_samples


# Set up 
DATA_PATH = os.path.join('..\\split_data_1\\right_view')
DATASET_PATH = os.path.join('D:\\SignLanguage\\merged')
LOG_PATH = os.path.join('..\\Logs')

video_folder = os.path.join(DATASET_PATH, 'right_view')
sequence_length = 60

os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

selected_actions = []

with open("..\\merged\\right_view.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

df = pd.DataFrame(metadata)
selected_actions = sorted(df['gloss'].unique())

label_map = {action: idx for idx, action in enumerate(selected_actions)}

label_map_path = os.path.join(LOG_PATH, 'label_map_2.json')

with open(label_map_path, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)


print(f"\n Selected {len(df['gloss'].unique())} actions.")

time = Gettime()
print(f"{datetime.now()} Start processing data...")


# Load the split data mapping by CSV files
train_df = pd.read_csv(os.path.join("..\\split_data", "train.csv"))
val_df = pd.read_csv(os.path.join("..\\split_data", "val.csv"))
test_df = pd.read_csv(os.path.join("..\\split_data", "test.csv"))


# main loop to process videos and create augmented samples
def process_and_augment_split(df_split, split_name, augment=False):
    print(f"\nProcessing {split_name} split with {len(df_split)} samples...")
    base_output_path = os.path.join(DATA_PATH, split_name)
    os.makedirs(base_output_path, exist_ok=True)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )as holistic:
        
        action_counter = {}

        action_position = {
            action: idx + 1 for idx, action in enumerate(pd.unique(df_split['gloss']))
        } 
        
      
        for action, group in df_split.groupby('gloss'):
            action_folder = create_folder(base_output_path, action)
            label = label_map[action]

            # init counter safely
            if action not in action_counter:
                existing_files = [f for f in os.listdir(action_folder) if f.endswith('.npz') if f.split('.')[0].isdigit()]
                indices = []
                for f in existing_files:
                    try:
                        indices.append(int(f.split('.')[0]))
                    except:
                        continue
                action_counter[action] = max(indices) + 1 if indices else 0
            
            current_total = len([f for f in os.listdir(action_folder) if f.endswith('.npz')])


            num_videos = len(group)

            if augment:
                TARGET = 200  

                if current_total >= TARGET:
                    base_aug = 0
                    remainder = 0
                    continue  # Skip augmentation if we already have enough samples
                else:
                    base_aug = (TARGET - num_videos) // num_videos
                    remainder = (TARGET - num_videos) % num_videos
            else:
                base_aug = 0
                remainder = 0

            for i, row in enumerate(group.itertuples()):
                video_file = f"{row.video_id:06d}.mp4"
                video_path = os.path.join(video_folder, video_file)


                if not os.path.exists(video_path):
                    print(f"Missing: {video_file}")
                    continue

                frame_lists = process_video(video_path, holistic)

                if not frame_lists:
                    continue

                # distribute remainder
                n_aug = base_aug + (1 if i < remainder else 0)

                if augment:
                    augmenteds = create_augmentation(frame_lists, augmentations, n_aug, 5)
                    augmenteds.append(frame_lists)
                else:

                    augmenteds = [frame_lists]

                for augmented in augmenteds:
                    seq = interpolate_keypoints(augmented)
                    if seq is None:
                        continue

                    idx = action_counter[action] 

                    file_path = os.path.join(action_folder, f"{idx}.npz")
                    np.savez(
                        file_path,
                        sequence=seq,
                        label=label
                    )

                    action_counter[action] += 1 

                    print(f"Action: {action_position[action]}/{len(df['gloss'].unique())} : {action} - Time: {time.get_time()}")

    print(f"{'-'*50}")
    print("Finished processing data.")




process_and_augment_split(train_df, "train", augment=True)
process_and_augment_split(val_df, "val", augment=False)
process_and_augment_split(test_df, "test", augment=False)
