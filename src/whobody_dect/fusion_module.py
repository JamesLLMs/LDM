import cv2
import mediapipe as mp
import numpy as np
import multiprocessing
import json
import copy
import os

# --- Standalone process worker function ---
def pose_worker(input_queue, output_queue):
    mp_pose = mp.solutions.pose
    # Enable world landmarks to get 3D coordinates in meters
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    while True:
        frame = input_queue.get()
        if frame is None: break  # Exit signal

        results = pose.process(frame)
        data = None
        if results.pose_world_landmarks:
            data = []
            for lm in results.pose_world_landmarks.landmark:
                data.append([lm.x, lm.y, lm.z])
        output_queue.put(data)

def hand_worker(input_queue, output_queue, selfie=False):
    mp_hands = mp.solutions.hands
    # If selfie/mirror mode, MediaPipe's labels are usually accurate;
    # If normal shooting, MediaPipe detects "Left" as the right side of the image (i.e., person's right hand).
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7
    )

    while True:
        frame = input_queue.get()
        if frame is None: break

        results = hands.process(frame)
        hands_data = {"Left": None, "Right": None}

        if results.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                # Core fix: get raw label
                raw_label = hand_handedness.classification[0].label

                # If not mirror mode (i.e., normal camera looking at others), we need to flip the label
                if not selfie:
                    actual_label = "Left" if raw_label == "Right" else "Right"
                else:
                    actual_label = raw_label

                if results.multi_hand_world_landmarks:
                    lm_list = results.multi_hand_world_landmarks[idx]
                    # Note: if the image is mirrored, x coordinate may also need to be negated, keep original 3D coordinates for now
                    coords = [[lm.x, lm.y, lm.z] for lm in lm_list.landmark]
                    hands_data[actual_label] = coords

        output_queue.put(hands_data)

class PoseHandFusion:
    def __init__(self, config_path="skeleton_config.json", is_selfie=False):

        # Locate current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_path)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Create ID to name mapping
        self.id_to_name = {}

        # Add Pose joint mappings
        for name, joint_id in self.config["pose_joints"].items():
            self.id_to_name[joint_id] = name

        # Add left hand joint mappings (ID = local_id + 100)
        left_offset = self.config["offsets"]["Left"]
        for name, local_id in self.config["hand_joints"].items():
            self.id_to_name[local_id + left_offset] = f"LEFT_HAND_{name}"

        # Add right hand joint mappings (ID = local_id + 200)
        right_offset = self.config["offsets"]["Right"]
        for name, local_id in self.config["hand_joints"].items():
            self.id_to_name[local_id + right_offset] = f"RIGHT_HAND_{name}"

        # Initialize multiprocessing
        self.pose_input = multiprocessing.Queue(maxsize=1)
        self.pose_output = multiprocessing.Queue(maxsize=1)
        self.hand_input = multiprocessing.Queue(maxsize=1)
        self.hand_output = multiprocessing.Queue(maxsize=1)

        self.p_pose = multiprocessing.Process(target=pose_worker, args=(self.pose_input, self.pose_output))
        self.is_selfie = is_selfie  # Mirror mode flag

        self.p_hand = multiprocessing.Process(target=hand_worker,args=(self.hand_input, self.hand_output, self.is_selfie)
        )
        
        self.p_pose.start()
        self.p_hand.start()

    def stop(self):
        self.pose_input.put(None)
        self.hand_input.put(None)
        self.p_pose.join()
        self.p_hand.join()

    def process_frame(self, frame):
        # 1. Distribute tasks (deep copy frame to avoid memory race issues, or rely on Copy-on-Write)
        # Note: Simplified direct put here, production may need shared memory optimization
        if self.pose_input.full(): self.pose_input.get()  # Clear old
        if self.hand_input.full(): self.hand_input.get()

        self.pose_input.put(frame)
        self.hand_input.put(frame)

        # 2. Wait for results (sync barrier)
        pose_res = self.pose_output.get()
        hands_res = self.hand_output.get()

        # 3. Data fusion
        return self.merge_data(pose_res, hands_res)

    def merge_data(self, pose_data, hands_data):
        """
        Core fusion logic:
        1. Load basic Pose data
        2. According to configuration rules, check if replacement is needed
        3. If hand exists, align hand data to Pose wrist position
        4. Rebuild connection table

        Returns:
        - joints_dict: {joint name: [x, y, z]}
        - connections_list: [[start name, end name], ...]
        """
        final_joints = {}  # {id: [x,y,z]}
        final_connections = []  # [[id1, id2], ...]

        # --- Step 1: Load human Pose data ---
        if pose_data:
            for i, coord in enumerate(pose_data):
                final_joints[i] = coord

        # --- Step 2: Load and align hand data ---
        # Offset mapping
        offsets = self.config["offsets"]

        # Temporarily store replaced mappings {old PoseID: new HandID}, used for updating connections
        id_remap_table = {}
        deleted_ids = set()

        # Process replacement rules
        for pose_id_str, rule in self.config["replacement_rules"].items():
            pose_id = int(pose_id_str)
            target_hand = rule.get("target_hand")

            # If rule is "delete": true, mark for deletion
            if rule.get("delete"):
                # Only when the corresponding hand is detected, delete the coarse hand node from Pose,
                # otherwise keep Pose's coarse node to prevent hand model from losing tracking
                if hands_data and hands_data.get(target_hand):
                    deleted_ids.add(pose_id)
                continue

            # If rule is replacement (override)
            if hands_data and hands_data.get(target_hand):
                hand_local_id = rule["target_joint_id"]
                global_hand_id = hand_local_id + offsets[target_hand]

                # Record mapping: this point from Pose becomes this point from hand
                id_remap_table[pose_id] = global_hand_id

                # Calculate coordinate alignment offset (Translation)
                # We assume the wrist position detected by Pose is the world coordinate anchor point of the hand model
                if pose_id in final_joints:
                    pose_wrist_pos = np.array(final_joints[pose_id])
                    # Get the raw coordinate of the corresponding wrist in the hand model (usually with origin at 0,0,0 or center of mass)
                    hand_wrist_raw = np.array(hands_data[target_hand][hand_local_id])

                    # Offset vector = Pose wrist - Hand wrist
                    translation_vector = pose_wrist_pos - hand_wrist_raw

                    # Apply this offset to all joints of this hand and add to final_joints
                    for h_idx, h_coord in enumerate(hands_data[target_hand]):
                        new_id = h_idx + offsets[target_hand]
                        aligned_pos = np.array(h_coord) + translation_vector
                        final_joints[new_id] = aligned_pos.tolist()

                    # Mark Pose's original wrist point as "replaced/deleted"
                    deleted_ids.add(pose_id)

                    # If rule explicitly specifies to delete Pose wrist, permanently delete
                    if rule.get("delete_pose_wrist"):
                        deleted_ids.add(pose_id)

        # --- Step 3: Filtering and cleanup ---
        # Apply filter configuration (config filters)
        for fid in self.config["filters"]:
            deleted_ids.add(fid)

        # Remove marked deleted points from results
        for del_id in deleted_ids:
            if del_id in final_joints:
                del final_joints[del_id]

        # --- Step 4: Dynamically rebuild connection relationships ---
        # 4.1 Add Pose connections
        for p_conn in self.config["pose_connections"]:
            start, end = p_conn

            # Check if endpoints are replaced (e.g., Pose Elbow -> Pose Wrist becomes Pose Elbow -> Hand Wrist)
            if start in id_remap_table: start = id_remap_table[start]
            if end in id_remap_table: end = id_remap_table[end]

            # Check if endpoints are deleted
            if start in deleted_ids or end in deleted_ids:
                continue  # If one side is deleted and not replaced, this line is broken

            # Only draw line when both points are in final joints
            if start in final_joints and end in final_joints:
                final_connections.append([start, end])

        # 4.2 Add Hand connections
        for hand_type in ["Left", "Right"]:
            if hands_data and hands_data.get(hand_type):
                offset = offsets[hand_type]
                # Only when this hand is actually added (i.e., replacement logic occurred or forcibly added)
                # Simple check: see if wrist ID exists in final_joints
                if (0 + offset) in final_joints:
                    for h_conn in self.config["hand_connections"]:
                        h_start = h_conn[0] + offset
                        h_end = h_conn[1] + offset
                        final_connections.append([h_start, h_end])

        # --- Step 5: Adjust hip position to be directly below shoulder ---
        # Ensure torso is vertical (rectangle formed by shoulder and hip is perpendicular to ground)
        LEFT_SHOULDER_ID = 11
        RIGHT_SHOULDER_ID = 12
        LEFT_HIP_ID = 23
        RIGHT_HIP_ID = 24

        # Adjust left hip: align x and z coordinates with left shoulder, keep y coordinate (height difference)
        if LEFT_SHOULDER_ID in final_joints and LEFT_HIP_ID in final_joints:
            shoulder_pos = final_joints[LEFT_SHOULDER_ID]
            hip_pos = final_joints[LEFT_HIP_ID]
            # Keep original y coordinate (height difference), but align x and z with shoulder
            final_joints[LEFT_HIP_ID] = [shoulder_pos[0], hip_pos[1], shoulder_pos[2]]

        # Adjust right hip: align x and z coordinates with right shoulder, keep y coordinate (height difference)
        if RIGHT_SHOULDER_ID in final_joints and RIGHT_HIP_ID in final_joints:
            shoulder_pos = final_joints[RIGHT_SHOULDER_ID]
            hip_pos = final_joints[RIGHT_HIP_ID]
            # Keep original y coordinate (height difference), but align x and z with shoulder
            final_joints[RIGHT_HIP_ID] = [shoulder_pos[0], hip_pos[1], shoulder_pos[2]]

        # --- Step 6: Convert IDs to names ---
        joints_dict = {}
        for joint_id, coords in final_joints.items():
            joint_name = self.id_to_name.get(joint_id, f"UNKNOWN_{joint_id}")
            joints_dict[joint_name] = coords

        connections_list = []
        for conn in final_connections:
            start_id, end_id = conn
            start_name = self.id_to_name.get(start_id, f"UNKNOWN_{start_id}")
            end_name = self.id_to_name.get(end_id, f"UNKNOWN_{end_id}")
            connections_list.append([start_name, end_name])

        return joints_dict, connections_list