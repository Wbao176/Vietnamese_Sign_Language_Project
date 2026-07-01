import numpy as np

FINGER_JOINTS = [
    (1, 2, 3, 4),  # Thumb
    (5, 6, 7, 8),  # Index
    (9, 10, 11, 12), # Middle
    (13, 14, 15, 16),# Ring
    (17, 18, 19, 20) # Pinky
]

WRIST_JOINTS = 0


# Compute the angle of there joints in the hand, which is the bend angle of each finger
def compute_finger_angle(a,b,c):
    vector1 = a - b
    vector2 = c - b

    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)

    return angle.astype(np.float32)

# Extract the finger angles from the hand keypoints
def extract_finger_angles(hand):
    hand = hand.reshape(21, 3)  # Ensure hand is in the shape (21, 3)
    angles = []
    for (a,b,c,d) in FINGER_JOINTS:
        angles.append(compute_finger_angle(hand[WRIST_JOINTS], hand[a], hand[b]))
        angles.append(compute_finger_angle(hand[a], hand[b], hand[c]))
        angles.append(compute_finger_angle(hand[b], hand[c], hand[d]))

    return np.array(angles, dtype=np.float32) / np.pi  # (15, )


# Compute the angle between two fingers, which is the inter-finger angle
def compute_inter_finger_angle(v1,v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # angle = np.arccos(cos_angle)
    angle = np.arccos(cos_angle)
    return angle.astype(np.float32)

# Extract the inter-finger distances from the hand keypoints
def extract_inter_finger_angles(hand):
    hand = hand.reshape(21, 3)  # hand (21, 3)
    vectors = [
        hand[2] - hand[1],  # Thumb
        hand[6] - hand[5],  # Index
        hand[10] - hand[9], # Middle
        hand[14] - hand[13],# Ring
        hand[18] - hand[17] # Pinky
    ]
    angles = []
    for i in range(len(vectors) - 1):
        angles.append(compute_inter_finger_angle(vectors[i], vectors[i + 1]))

    return np.array(angles, dtype=np.float32) / np.pi # (4, )


def extract_hand_shape(hand):
    P = extract_finger_angles(hand)        # (15,)
    C = extract_inter_finger_angles(hand)  # (4,)
    H  = np.concatenate([P, C])             # (19,)
    return H    


# Extract palm orientation (by normal vector)
def compute_palm_orientation(v1, v2):

    normal_vector = np.cross(v1, v2)  # Normal vector of the palm plane
    normal_vector /= np.linalg.norm(normal_vector) + 1e-6  # Normalize the normal vector

    return normal_vector  # (3,)

def extract_palm_orientation(hand):
    hand = hand.reshape(21, 3)  # hand (21, 3)
    v1 = hand[5] - hand[0]  # Vector from wrist to index MCP
    v2 = hand[17] - hand[0] # Vector from wrist to pinky MCP
    n_vector = compute_palm_orientation(v1, v2)

    return np.array(n_vector, dtype=np.float32)  # (3,)


# Extract hand location (middle finger [9])
def extract_hand_location(hand):
    hand = hand.reshape(21, 3)
    
    l = hand[9] - hand[WRIST_JOINTS]  # middle finger MCP
    
    return l.astype(np.float32)  # (3,)


def compute_velocity(p_prev, p_curr):
    velocity = (p_curr - p_prev).astype(np.float32)
    return np.clip(velocity, -1.0, 1.0)  # (3,)

def compute_acceleration(p_prev, p_curr, p_next):
    acceleration = (p_next - 2 * p_curr + p_prev).astype(np.float32)
    return np.clip(acceleration, -0.5, 0.5)

def compute_curvature(p_prev, p_curr, p_next):
    v1 = p_curr - p_prev
    v2 = p_next - p_curr

    area = 0.5 * np.linalg.norm(np.cross(v1, v2))

    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)
    d3 = np.linalg.norm(p_next - p_prev)

    denominator = d1 * d2 * d3 + 1e-6
    curvature = (4.0 * area) / denominator

    curvature = np.clip(curvature, 0, 10.0)
    log_curvature = np.log1p(curvature)  # Logarithmic scaling
    normalized_curvature = log_curvature / 2.4 # Normalize to [0, 1] based on the expected range of curvature

    return np.array([normalized_curvature], dtype=np.float32)

def extract_motion_features(hand_sequence):
    motion_feature = []
    
    for i in range(1, len(hand_sequence) - 1):
        p_prev = hand_sequence[i - 1]
        p_curr = hand_sequence[i]
        p_next = hand_sequence[i + 1]

        velocity = compute_velocity(p_prev, p_curr)
        acceleration = compute_acceleration(p_prev, p_curr, p_next)
        curvature = compute_curvature(p_prev, p_curr, p_next)

        motion_feature.append(np.concatenate([velocity, acceleration, curvature]))

    return np.array(motion_feature, dtype=np.float32)  # (T-2, 7)


def get_hand_trajectory(sequence):
    left_hand_traj = []
    right_hand_traj = []

    first_l = next((frame[75:138].reshape(21, 3)[9] for frame in sequence if np.any(frame[75:138] != 0)), np.zeros(3))
    first_r = next((frame[138:201].reshape(21, 3)[9] for frame in sequence if np.any(frame[138:201] != 0)), np.zeros(3))
    last_l = first_l
    last_r = first_r

    for frame in sequence:
        left_hand = frame[75:138].reshape(21, 3)
        right_hand = frame[138:201].reshape(21, 3)

        if np.any(left_hand != 0):
            last_l = left_hand[9]  # Update last known position if current frame has valid left hand
        left_hand_traj.append(last_l)
            
        if np.any(right_hand != 0):
            last_r = right_hand[9]  # Update last known position if current frame has valid right hand
        right_hand_traj.append(last_r)

    return (np.array(left_hand_traj, dtype=np.float32), np.array(right_hand_traj, dtype=np.float32))  # (60, 3), (60, 3)

def extract_motion_features_from_sequence(sequence):
    left_traj, right_traj = get_hand_trajectory(sequence)  # (60, 3), (60, 3)

    left_motion = extract_motion_features(left_traj)  # (58, 7)
    right_motion = extract_motion_features(right_traj)  # (58, 7)

    left_motion = np.vstack([left_motion[0], left_motion, left_motion[-1]])  # Pad to (60, 7)
    right_motion = np.vstack([right_motion[0], right_motion, right_motion[-1]])  # Pad to (60, 7)
    return np.concatenate([left_motion, right_motion], axis=1)  # (60, 14)