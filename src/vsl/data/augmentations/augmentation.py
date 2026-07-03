import numpy as np
import mediapipe as mp
import random
import os
import math

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


# Scale function
def scale_sequence_function(
        keypoint_sequences,  # Input Sequences
        scale_factor_range = (0.7, 1.26),  # Scaling range (can be change) --should not be zero or negative
        total_landmarks = TOTAL_LANDMARKS,
        landmarks_center = UPPER_BODY_POSE_LANDMARKS - 2,
        norm_01 = True  # Normalize allow keypoints (x, y) to [0, 1] after scale
):
        processed_keypoints = []
        if not keypoint_sequences:
             return processed_keypoints
        
        current_scale_factor = random.uniform(scale_factor_range[0], scale_factor_range[1])

        
        # Defensive programming (code it if have time)
        # if scale_factor_range <= 0:
        # ...

        for frame_keypoint_flattened in keypoint_sequences:
                if frame_keypoint_flattened is None:
                  processed_keypoints.append(None)
                  continue
                if not isinstance(frame_keypoint_flattened, np.ndarray) or frame_keypoint_flattened.shape != (total_landmarks *3,):
                  processed_keypoints.append(frame_keypoint_flattened.copy())
                  continue
                try: 
                        # Reshape:
                        # 0:      [x1, y1, z1]
                        # 1:      [x2, y2, z2]
                        #...:          ...
                        current_points = frame_keypoint_flattened.copy().reshape(total_landmarks, 3) 
                except ValueError:
                        processed_keypoints.append(frame_keypoint_flattened.copy())
                        continue

                # Calculate for center scale
                pose_points = current_points[0:UPPER_BODY_POSE_LANDMARKS] # (0:25) take all [x, y, z]
                hand_idx_begin = UPPER_BODY_POSE_LANDMARKS
                hand_points_to_scale = current_points[hand_idx_begin:] 
                center_points_pose = pose_points[0:landmarks_center]
                center_points_pose_list = [center_points_pose]
                if hand_points_to_scale.shape[0] > 0: # if have shape add the hands to list to center
                     center_points_pose_list.append(hand_points_to_scale)
                
                center_x, center_y = 0.0, 0.0 # set point as default
                calculate_center = False # set false as default

                if center_points_pose_list:
                     center_points_pose_concat = np.concatenate(center_points_pose_list, axis=0)
                     
                     valid_center_points_mask = np.any(center_points_pose_concat != 0, axis=1)
                     valid_center_points = center_points_pose_concat[valid_center_points_mask]

                     if valid_center_points.shape[0] > 0: 
                          center_x = np.median(valid_center_points[:, 0])
                          center_y = np.median(valid_center_points[:, 1])
                          calculate_center = True
                     else:
                          all_valid_points_mask = np.any(current_points != 0, axis=1)
                          all_valid_points = current_points[all_valid_points_mask]
                          if all_valid_points.shape[0] > 0:
                               center_x = np.median(all_valid_points[:, 0])
                               center_y = np.median(all_valid_points[:, 1])
                               calculate_center = True
                # Apply scale               
                if calculate_center:
                     all_valid_points_mask_for_scaling = np.any(current_points != 0, axis=1)
                     if np.any(all_valid_points_mask_for_scaling):
                          x_all_valid = current_points[all_valid_points_mask_for_scaling, 0]
                          y_all_valid = current_points[all_valid_points_mask_for_scaling, 1]

                          x_trans = x_all_valid - center_x
                          y_trans = y_all_valid - center_y
                          x_scale = x_trans * current_scale_factor
                          y_scale = y_trans * current_scale_factor
                          new_x = x_scale + center_x
                          new_y = y_scale + center_y
                          current_points[all_valid_points_mask_for_scaling, 0] = new_x
                          current_points[all_valid_points_mask_for_scaling, 1] = new_y

                if norm_01:
                     valid_mask_xy = np.any(current_points[:, :2] != 0, axis=1)

                     if np.any(valid_mask_xy):
                          x_coords = current_points[valid_mask_xy, 0]
                          y_coords = current_points[valid_mask_xy, 1]
                          
                          x_min, x_max = np.min(x_coords), np.max(x_coords)
                          y_min, y_max = np.min(y_coords), np.max(y_coords)

                        # Normalize for X
                          if (x_max - x_min) > 1e-7: # max - min may be close to zero but if not greater than 1e-7, treat it as 0
                                current_points[valid_mask_xy, 0] = (x_coords - x_min) / (x_max - x_min)
                          elif (x_coords.size > 0): # if 
                               current_points[valid_mask_xy, 0] = 0.5 

                        # Normalize for Y
                          if (y_max - y_min) > 1e-7: # max - min may be close to zero but if not greater than 1e-7, treat it as 0
                                current_points[valid_mask_xy, 1] = (y_coords - y_min) / (y_max - y_min)
                          elif (y_coords.size > 0): # if 
                               current_points[valid_mask_xy, 1] = 0.5 


                scale_frame_flatten = current_points.flatten()

                if np.isnan(scale_frame_flatten).any() or np.isinf(scale_frame_flatten).any():
                     processed_keypoints.append(frame_keypoint_flattened.copy()) # Keep original frame if NaN or INf
                else:
                     processed_keypoints.append(scale_frame_flatten)

        return processed_keypoints



# Rotate sequence 
def rotate_sequence_function(
        keypoint_sequence,
        rotate_factor = (-15.0, 15.0), # this is deg
        total_landmarks = TOTAL_LANDMARKS,
        pose_center_landmarks = UPPER_BODY_POSE_LANDMARKS - 2
):
        rotate_keypoint = []
        if not keypoint_sequence:
                return rotate_keypoint
        
        angle_deg = random.uniform(rotate_factor[0], rotate_factor[1])

        angle_rad = math.radians(angle_deg) # shift it to rad to easily calculate sin/cos angle
        cos_angle = math.cos(angle_rad) # cos angle
        sin_angle = math.sin(angle_rad) # sin angle
        
        for frame_keypoint_flatten in keypoint_sequence:
             if frame_keypoint_flatten is None:
                rotate_keypoint.append(None)
             
             if not isinstance(frame_keypoint_flatten, np.ndarray) or frame_keypoint_flatten.shape[0] != (total_landmarks * 3,):
                rotate_keypoint.append(frame_keypoint_flatten.copy())
                continue
             
             try:
                  current_point = frame_keypoint_flatten.copy().reshape(total_landmarks, 3)
             except ValueError:
                  rotate_keypoint.append(frame_keypoint_flatten.copy())
                  continue
             

             pose_points = current_point[0:UPPER_BODY_POSE_LANDMARKS]
             hand_idx_begin = UPPER_BODY_POSE_LANDMARKS
             hand_points = current_point[hand_idx_begin:]
             center_points_pose = pose_points[0:pose_center_landmarks]
             center_points_pose_list = [center_points_pose]

             if hand_points.shape[0] > 0:
                  center_points_pose_list.append(hand_points)

             
             center_x, center_y = 0.0, 0.0
             calculate_center = False

             if center_points_pose_list:
                  center_points_pose_concat = np.concatenate(center_points_pose_list, axis=0)
                  valid_center_points_mask = np.any(center_points_pose_concat != 0, axis=1)
                  valid_center_points = center_points_pose_concat[valid_center_points_mask]

                  if valid_center_points.shape[0] > 0:
                       center_x = np.median(valid_center_points[:, 0])
                       center_y = np.median(valid_center_points[:, 1])
                       calculate_center = True
                  else:
                       all_valid_center_masks = np.any(current_point != 0, axis=1)
                       all_valid_center_points = current_point[all_valid_center_masks]

                       if all_valid_center_points.shape[0] > 0:
                                center_x = np.median(all_valid_center_points[:, 0])
                                center_y = np.median(all_valid_center_points[:, 1])
                                calculate_center = True

             if calculate_center:
                   all_valid_masks_for_rotate = np.any(current_point != 0, axis=1)
                   if np.any(all_valid_masks_for_rotate):
                        x_valid = current_point[all_valid_masks_for_rotate, 0]
                        y_valid = current_point[all_valid_masks_for_rotate, 1]

                        x_trans = x_valid - center_x
                        y_trans = y_valid - center_y

                        # rorate
                        x_rotate = x_trans * cos_angle - y_trans * sin_angle
                        y_rotate = x_trans * sin_angle + y_trans * cos_angle

                        x_new = x_rotate + center_x
                        y_new = y_rotate + center_y

                        rotate_current_points = current_point.copy()

                        rotate_current_points[all_valid_masks_for_rotate, 0] = x_new
                        rotate_current_points[all_valid_masks_for_rotate, 1] = y_new

                        rotate_frame_flatten = rotate_current_points.flatten()

                        if np.isnan(rotate_frame_flatten).any() or np.isinf(rotate_frame_flatten).any():
                             rotate_keypoint.append(frame_keypoint_flatten.copy())
                        else:
                             rotate_keypoint.append(rotate_frame_flatten)

        return rotate_keypoint


# Translate Keypoint (Translate randomly in (dx, dy) 2 dimension)

def translate_sequence_function(
     keypoint_sequence,
     translate_x_range = (-0.05, 0.05),  # min / max for translate on x range
     translate_y_range = (-0.05, 0.05),  # min / max for translate on y range
     clip_01: bool = True,
     total_landmark = TOTAL_LANDMARKS

):
     translated_sequence = []
     if not keypoint_sequence:
          return translated_sequence
     
     curruent_translate_x = random.uniform(translate_x_range[0], translate_x_range[1])
     current_translate_y = random.uniform(translate_y_range[0], translate_y_range[1])
     

     for frame_keypoint_flatten in keypoint_sequence:
          if frame_keypoint_flatten is None:
               translated_sequence.append(None)
               continue

          if not isinstance (frame_keypoint_flatten, np.ndarray) or frame_keypoint_flatten.shape != (total_landmark *3,):
               translated_sequence.append(frame_keypoint_flatten.copy())
               continue
          
          try: 
               current_points = frame_keypoint_flatten.copy().reshape(total_landmark, 3)

          except ValueError:
               translated_sequence.append(frame_keypoint_flatten.copy())
               continue

          translated_current_point = current_points.copy()

          valid_mask = np.any(current_points != 0, axis=1)

          translated_current_point[valid_mask, 0] += curruent_translate_x # x_new
          translated_current_point[valid_mask, 1] += current_translate_y # y_new

          if clip_01:
               translated_current_point[valid_mask, 0] = np.clip(translated_current_point[valid_mask, 0], 0.0, 1.0)
               translated_current_point[valid_mask, 1] = np.clip(translated_current_point[valid_mask, 1], 0.0, 1.0)


          translated_frame_flatten = translated_current_point.flatten()

          if np.isnan(translated_frame_flatten).any() or np.isinf(translated_frame_flatten).any():
               translated_sequence.append(frame_keypoint_flatten.copy())
          else:
               translated_sequence.append(translated_frame_flatten)

     return translated_sequence



# Time / Speed Function
def time_stretch_function(
     keypoints_sequence,
     speed_factor_range = (0.8, 1.2),  # 0.8 is slower than 20% and 1.2 is faster than 20%          
):
     
     stretch_sequence = []
     if not keypoints_sequence or all(kp is None for kp in keypoints_sequence):
          return stretch_sequence # return empty list if input is empty or all frames are None
     
     # Sort out frame that not None to process
     valid_frames = [kp for kp in keypoints_sequence if kp is not None]
     if not valid_frames:
          return keypoints_sequence
     
     org_num_valid_frames = len(valid_frames)

     current_speed_factor = random.uniform(speed_factor_range[0], speed_factor_range[1]) # Randomly select a speed factor within the specified range

     if current_speed_factor == 1:  # No change
          return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence] #return a copy of the original sequence if speed factor is 1 (no change)
     

     new_num_valid_frames = int(round(org_num_valid_frames / current_speed_factor)) # Time = Distance / Speed, so new time (number of frames) = original time (number of frames) / speed factor
     if new_num_valid_frames == 0: # Avoid having zero frames after stretching
          if org_num_valid_frames > 0: # If there are valid frames, keep at least one frame
               stretch_sequence.append(valid_frames[0].copy() if valid_frames[0] is not None else None)
          
          return stretch_sequence 
     
     org_indices = np.linspace(0, org_num_valid_frames - 1, new_num_valid_frames) # Generate new indices for the stretched sequence

     resampled_indices = np.round(org_indices).astype(int) # Round the indices to the nearest integer to get valid frame indices
     resampled_indices = np.clip(resampled_indices, 0, org_num_valid_frames - 1) # Ensure indices are within the valid range
     
     for idx in resampled_indices:     
          stretch_sequence.append(valid_frames[idx].copy() if valid_frames[idx] is not None else None) # Append the corresponding frame from the original sequence based on the resampled indices, ensuring to copy the frame if it's a numpy array to avoid modifying the original data
     return stretch_sequence



# Define the indices for upper body landmarks (based on MediaPipe Holistic)
L_SHOULDER_IDX = 11
R_SHOULDER_IDX = 12
L_ELBOW_IDX = 13
R_ELBOW_IDX = 14
L_WRIST_IDX = 15
R_WRIST_IDX = 16

def inverse_kinematics_function(
     shoulder_xy : np.ndarray,
     wrist_target_xy : np.ndarray,
     len_arm : float,
     len_forearm : float,
     original_elbow_xy : None,
     original_wrist_xy : None,
     prefer_original_bend: bool = True
):
     
     distance_shoulder_to_target = np.linalg.norm(wrist_target_xy - shoulder_xy)
     len_1 = max(1e-7, len_arm)  # Avoid division by zero
     len_2 = max(1e-7, len_forearm)  # Avoid division by zero

     # Case 1: Wrist target is out of reach (further than the arm can extend)
     if distance_shoulder_to_target > len_1 + len_2 - 1e-5: 
          if distance_shoulder_to_target < 1e-5:  # Target is extremely close to shoulder
               return shoulder_xy + np.array([len_1, 0]) if original_elbow_xy is None else original_elbow_xy.copy()
          vector_pointing_wrist = (wrist_target_xy - shoulder_xy) / distance_shoulder_to_target # Unit vector from shoulder to target
          elbow_xy = shoulder_xy + vector_pointing_wrist * len_1 # Elbow is fully extended towards the target
          return elbow_xy
     
     # Case 2: Wrist target is too close (closer than the arm can bend)
     if distance_shoulder_to_target < abs(len_1-len_2) + 1e-5:
          if original_elbow_xy is not None:
               return original_elbow_xy.copy() # Elbow position is determined by the original bend direction
          
          if distance_shoulder_to_target < 1e-5:  # Target is extremely close to shoulder
               return shoulder_xy + np.array([len_1, 0]) # Elbow is fully extended in a default direction (e.g., to the right)
          
          vector_pointing_wrist = (wrist_target_xy - shoulder_xy) / distance_shoulder_to_target # Unit vector from shoulder to target
          elbow_xy = shoulder_xy + vector_pointing_wrist * len_1 # Elbow is fully extended towards the target
          return elbow_xy
     
     # Case 3: Wrist target is reachable (within the arm's range of motion)
     if distance_shoulder_to_target < 1e-5:  distance_shoulder_to_target = 1e-5  # Avoid division by zero in calculations

     # Distance from shoulder to projection point on the line connecting shoulder and wrist target
     a = (distance_shoulder_to_target**2 + len_1**2 - len_2**2) / (2 * distance_shoulder_to_target) # Distance from shoulder to the line connecting shoulder and wrist target where the elbow is located
     h_squared = len_1**2 - a**2  # Square of the height from the line connecting shoulder and wrist target to the elbow position

     if h_squared < -1e-9:
          vector_pointing_wrist = (wrist_target_xy - shoulder_xy) / distance_shoulder_to_target # Unit vector from shoulder to target
          elbow_xy = shoulder_xy + vector_pointing_wrist * len_1 # Elbow is fully extended towards the target
          return elbow_xy
     
     h = np.sqrt(max(0, h_squared))  # Height from the line connecting shoulder and wrist target to the elbow position (ensure non-negative)


     projection_point_x = shoulder_xy[0] + a * (wrist_target_xy[0] - shoulder_xy[0]) / distance_shoulder_to_target
     projection_point_y = shoulder_xy[1] + a * (wrist_target_xy[1] - shoulder_xy[1]) / distance_shoulder_to_target

     perpendicular_vector_x = - (wrist_target_xy[1] - shoulder_xy[1]) / distance_shoulder_to_target
     perpendicular_vector_y = (wrist_target_xy[0] - shoulder_xy[0]) / distance_shoulder_to_target

     elbow_sol1_xy = np.array([projection_point_x + h * perpendicular_vector_x, projection_point_y + h * perpendicular_vector_y])
     elbow_sol2_xy = np.array([projection_point_x - h * perpendicular_vector_x, projection_point_y - h * perpendicular_vector_y])

     # Choose the solution that is closer to the original elbow position if prefer_original_bend is True and original_elbow_xy is provided
     if prefer_original_bend and original_elbow_xy is not None and original_wrist_xy is not None:
          if original_elbow_xy is None or original_wrist_xy is None:
               dis_1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
               dis_2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
               return elbow_sol1_xy if dis_1 <= dis_2 else elbow_sol2_xy
          
          return elbow_sol1_xy
     
     vector_Shoulder_wrist = original_wrist_xy - shoulder_xy
     if np.linalg.norm(vector_Shoulder_wrist) < 1e-5:  # If the original wrist position is extremely close to the shoulder, we cannot determine the bend direction based on the original pose
          dis_1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
          dis_2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)

          return elbow_sol1_xy if dis_1 <= dis_2 else elbow_sol2_xy
     
     original_size = (original_wrist_xy[0]- shoulder_xy[0]) * (original_elbow_xy[0] - shoulder_xy[0]) - (original_wrist_xy[1]- shoulder_xy[1]) * (original_elbow_xy[1] - shoulder_xy[1]) # Calculate the original cross product to determine the original bend direction

     side_solution_1 = (wrist_target_xy[0]- shoulder_xy[0]) * (elbow_sol1_xy[0] - shoulder_xy[0]) - (wrist_target_xy[1]- shoulder_xy[1]) * (elbow_sol1_xy[1] - shoulder_xy[1]) # Calculate the cross product for solution 1
     side_solution_2 = (wrist_target_xy[0]- shoulder_xy[0]) * (elbow_sol2_xy[0] - shoulder_xy[0]) - (wrist_target_xy[1]- shoulder_xy[1]) * (elbow_sol2_xy[1] - shoulder_xy[1]) # Calculate the cross product for solution 2

     if abs(original_size) < 1e-3:  # If the original bend direction is ambiguous, choose the solution that is closer to the original elbow position\
          dis_1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
          dis_2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
          
          return elbow_sol1_xy if dis_1 <= dis_2 else elbow_sol2_xy
     
     if np.sign(original_size) == np.sign(side_solution_1):
          return elbow_sol1_xy
     elif np.sign(original_size) == np.sign(side_solution_2):
          return elbow_sol2_xy
     else:
          dis_1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
          dis_2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
          
          return elbow_sol1_xy if dis_1 <= dis_2 else elbow_sol2_xy
     

def inter_hand_distance_function(
     keypoint_sequence,
     dx_change_range = (-0.1, 0.1),
     dy_shift_range = (-0.03, 0.03),
     clip_01: bool = True,
     total_landmarks = TOTAL_LANDMARKS,
):
     augmented_sequence = []
     if not keypoint_sequence:
          return augmented_sequence
     
     current_change_dx = random.uniform(dx_change_range[0], dx_change_range[1])  # Randomly select a change in distance between hands within the specified range
     current_shift_dy = random.uniform(dy_shift_range[0], dy_shift_range[1])  # Randomly select a shift in y direction for the hands within the specified range

     for frame_keypoint_flatten in keypoint_sequence:
          if frame_keypoint_flatten is None:
               augmented_sequence.append(None)
               continue
          if not isinstance(frame_keypoint_flatten, np.ndarray) or frame_keypoint_flatten.shape != (total_landmarks * 3,):
               augmented_sequence.append(frame_keypoint_flatten.copy())
               continue
          try:
               all_origin_points = frame_keypoint_flatten.copy().reshape(total_landmarks, 3)
          except ValueError:
               augmented_sequence.append(frame_keypoint_flatten.copy())
               continue

          augmented_points = all_origin_points.copy() # Start with the original frame as the base for augmentation


          # Extracrt the original positions of the relevant landmarks for both hands (shoulders, elbows, wrists)

          original_left_shoulder_xy = all_origin_points[L_SHOULDER_IDX, 0:2].copy()
          original_left_elbow_xy = all_origin_points[L_ELBOW_IDX, 0:2].copy()
          original_left_wrist_xy = all_origin_points[L_WRIST_IDX, 0:2].copy()

          original_right_shoulder_xy = all_origin_points[R_SHOULDER_IDX, 0:2].copy()
          original_right_elbow_xy = all_origin_points[R_ELBOW_IDX, 0:2].copy()
          original_right_wrist_xy = all_origin_points[R_WRIST_IDX, 0:2].copy()

          left_side_valid = np.all(original_left_shoulder_xy != 0) and np.all(original_left_elbow_xy != 0) and np.all(original_left_wrist_xy != 0)
          right_side_valid = np.all(original_right_shoulder_xy != 0) and np.all(original_right_elbow_xy != 0) and np.all(original_right_wrist_xy != 0)

          if np.all(original_left_wrist_xy != 0) and np.all(original_right_wrist_xy != 0):
               curent_mid_point_dx = (original_left_wrist_xy[0] + original_right_wrist_xy[0]) / 2

               x_left = min(original_left_wrist_xy[0], original_right_wrist_xy[0])
               x_right = max(original_left_wrist_xy[0], original_right_wrist_xy[0])
               current_distance = x_right - x_left

               target_distance = current_distance + current_change_dx

               if target_distance < 1e-3:  # Avoid division by zero or negative distance
                    target_distance = 1e-3
               
               if original_left_wrist_xy[0] <= original_right_wrist_xy[0]:  # Left wrist is on the left side
                    target_left_wrist_x = curent_mid_point_dx - target_distance / 2
                    target_right_wrist_x = curent_mid_point_dx + target_distance / 2

               else:  # Right wrist is on the left side
                    target_right_wrist_x = curent_mid_point_dx + target_distance / 2
                    target_left_wrist_x = curent_mid_point_dx - target_distance / 2

          else: # If wrist positions are not valid, we cannot determine the current distance or mid-point, so we will just apply a random change to the original wrist positions without considering the current distance
               target_left_wrist_x = original_left_wrist_xy[0]
               target_right_wrist_x = original_right_wrist_xy[0]


          # Process left hand:
          if left_side_valid:
               len_left_arm = np.linalg.norm(original_left_elbow_xy - original_left_shoulder_xy)
               len_left_forearm = np.linalg.norm(original_left_wrist_xy - original_left_elbow_xy)

               target_left_wrist_xy_for_ik = np.array([target_left_wrist_x, original_left_wrist_xy[1]])

               new_left_elbow_xy = inverse_kinematics_function(original_left_shoulder_xy, target_left_wrist_xy_for_ik, len_left_arm, len_left_forearm, original_left_elbow_xy, original_left_wrist_xy)
               
               if new_left_elbow_xy is not None:
                    left_wrist_dx = target_left_wrist_xy_for_ik[0] - original_left_wrist_xy[0]
                    left_wrist_dy = target_left_wrist_xy_for_ik[1] - original_left_wrist_xy[1]

                    augmented_points[L_ELBOW_IDX, 0:2] = new_left_elbow_xy
                    augmented_points[L_WRIST_IDX, 0] = target_left_wrist_xy_for_ik[0]

                    idx_left_hand_start = UPPER_BODY_POSE_LANDMARKS
                    idx_left_hand_end = UPPER_BODY_POSE_LANDMARKS + LEFT_HAND_LANDMARKS
                    left_hand_part = augmented_points[idx_left_hand_start:idx_left_hand_end]
                    valid_left_hand_mask = np.any(left_hand_part != 0, axis=1)
                    
                    if np.any(valid_left_hand_mask):
                         left_hand_part[valid_left_hand_mask, 0] += left_wrist_dx
                         left_hand_part[valid_left_hand_mask, 1] += left_wrist_dy
                    augmented_points[idx_left_hand_start:idx_left_hand_end] = left_hand_part

          
          # Process right hand:
          if right_side_valid:
               len_right_arm = np.linalg.norm(original_right_elbow_xy - original_right_shoulder_xy)
               len_right_forearm = np.linalg.norm(original_right_wrist_xy - original_right_elbow_xy)

               target_right_wrist_xy_for_ik = np.array([target_right_wrist_x, original_right_wrist_xy[1]])

               new_right_elbow_xy = inverse_kinematics_function(original_right_shoulder_xy, target_right_wrist_xy_for_ik, len_right_arm, len_right_forearm, original_right_elbow_xy, original_right_wrist_xy)

               if new_right_elbow_xy is not None:
                    right_wrist_dx = target_right_wrist_xy_for_ik[0] - original_right_wrist_xy[0]
                    right_wrist_dy = target_right_wrist_xy_for_ik[1] - original_right_wrist_xy[1]

                    augmented_points[R_ELBOW_IDX, 0:2] = new_right_elbow_xy
                    augmented_points[R_WRIST_IDX, 0] = target_right_wrist_xy_for_ik[0]

                    idx_right_hand_start = UPPER_BODY_POSE_LANDMARKS + LEFT_HAND_LANDMARKS
                    idx_right_hand_end = UPPER_BODY_POSE_LANDMARKS + LEFT_HAND_LANDMARKS + RIGHT_HAND_LANDMARKS
                    right_hand_part = augmented_points[idx_right_hand_start:idx_right_hand_end]
                    valid_right_hand_mask = np.any(right_hand_part != 0, axis=1)

                    if np.any(valid_right_hand_mask):
                         right_hand_part[valid_right_hand_mask, 0] += right_wrist_dx
                         right_hand_part[valid_right_hand_mask, 1] += right_wrist_dy
                    augmented_points[idx_right_hand_start:idx_right_hand_end] = right_hand_part

          # Apply shift in y direction for both hands
          if abs(current_shift_dy) > 1e-5:
               arm_and_hand_indices = [L_SHOULDER_IDX, L_ELBOW_IDX, L_WRIST_IDX, R_SHOULDER_IDX, R_ELBOW_IDX, R_WRIST_IDX]
               arm_and_hand_indices.extend(list(range(UPPER_BODY_POSE_LANDMARKS, TOTAL_LANDMARKS)))  # Add hand landmark indices to the list

               unique_arm_and_hand_indices = sorted(list(set(arm_and_hand_indices)))  # Ensure unique indices in case of any duplicates

               for idx in unique_arm_and_hand_indices:
                    is_left_arm_part = (idx == L_WRIST_IDX or idx == L_ELBOW_IDX)
                    is_right_arm_part = (idx == R_WRIST_IDX or idx == R_ELBOW_IDX)
                    is_left_hand_part = (idx >= UPPER_BODY_POSE_LANDMARKS and idx < UPPER_BODY_POSE_LANDMARKS + LEFT_HAND_LANDMARKS)
                    is_right_hand_part = (idx >= UPPER_BODY_POSE_LANDMARKS + LEFT_HAND_LANDMARKS and idx < total_landmarks)  

                    should_shift_y = (is_left_arm_part and left_side_valid) or \
                                   (is_right_arm_part and right_side_valid) or \
                                   (is_left_hand_part and left_side_valid) or \
                                   (is_right_hand_part and right_side_valid)
                    if should_shift_y and idx < len(augmented_points) and np.any(augmented_points[idx, 0:2] !=0):
                         augmented_points[idx, 1] += current_shift_dy

          # Clip
          if clip_01:
               indices_to_clip = list(range(L_SHOULDER_IDX, total_landmarks)) # Từ vai trở đi
               for idx in indices_to_clip:
                    if idx < len(augmented_points) and np.any(augmented_points[idx,0:2] !=0): # Chỉ clip điểm XY khác 0
                         augmented_points[idx, 0] = np.clip(augmented_points[idx, 0], 0.0, 1.0)
                         augmented_points[idx, 1] = np.clip(augmented_points[idx, 1], 0.0, 1.0)

          augmented_frame_flat_output = augmented_points.flatten()
          if np.isnan(augmented_frame_flat_output).any() or np.isinf(augmented_frame_flat_output).any():
               augmented_sequence.append(frame_keypoint_flatten.copy())
          else:
               augmented_sequence.append(augmented_frame_flat_output)
     return augmented_sequence