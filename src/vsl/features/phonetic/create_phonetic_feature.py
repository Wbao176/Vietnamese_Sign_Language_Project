import numpy as np
import glob, os
from notebooks.archive.phonetic_feature.phonetic_extraction import extract_hand_shape, extract_palm_orientation, extract_hand_location, extract_motion_features_from_sequence

DATA_PATH = "..\\split_data\\right_view"
OUTPUT_PATH = "..\\phonetic\\right_view" 

os.makedirs(OUTPUT_PATH, exist_ok=True)

def process_sequence(sequence):
    T = len(sequence)

    motion = extract_motion_features_from_sequence(sequence)

    new_seq = []

    for i in range(T):
        frame = sequence[i]
        left_hand = frame[75:138]  # (63,)
        right_hand = frame[138:201] # (63,)

        if np.all(left_hand) == 0:
            s_l = np.zeros(19)
            n_l = np.zeros(3)
            l_l = np.zeros(3)
        else:
            s_l = extract_hand_shape(left_hand)          # (19,)
            n_l = extract_palm_orientation(left_hand)    # (3,)
            l_l = extract_hand_location(left_hand)       # (3,)

        if np.all(right_hand) == 0:
            s_r = np.zeros(19)
            n_r = np.zeros(3)
            l_r = np.zeros(3)
        else:
            s_r = extract_hand_shape(right_hand)          # (19,)
            n_r = extract_palm_orientation(right_hand)    # (3,)
            l_r = extract_hand_location(right_hand)       # (3,)


        m = motion[i]                         # (7,)

        new_frame = np.concatenate([
            s_l, n_l, l_l, s_r, n_r, l_r, m
        ])  # (19 + 3 + 3) * 2 + 7*2 = 64


        new_seq.append(new_frame)

    return np.array(new_seq, dtype=np.float32)


splits = ['train', 'val', 'test']

for split in splits:
    input_path = os.path.join(DATA_PATH, split)
    output_path = os.path.join(OUTPUT_PATH, split)

    files = glob.glob(os.path.join(input_path, '**', '*.npz'), recursive=True)

    print(f"Processing {split} : {len(files)} files")

    for file in files:
        data  = np.load(file)

        sequence = data['sequence']
        label = data['label']  

        new_seq = process_sequence(sequence)

        rel_path = os.path.relpath(file, input_path)
        save_file = os.path.join(output_path, rel_path)

        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        np.savez(save_file, sequence=new_seq, label=label)