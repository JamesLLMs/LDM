"""
Arm Pose API Module

Provides a simple interface to get fused coordinates of both arms and hands (URDF standard compliant).

================================================================================
Length Normalization Functionality
================================================================================

This module uses standard joint lengths defined in human_config.json to perform length normalization on detection results.
This ensures that joint distances in URDF output remain fixed and do not change due to viewing angle variations.

Length normalization method:
1. Calculate current distance: d = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
2. Apply scaling factor: scale = target_length / current_length
3. Scale joint coordinates: new_coord = orig_coord * scale

================================================================================
Return Value Description
================================================================================

get_arm_coordinates(frame) returns:
    (left_arm, right_arm) - two dictionaries

Each dictionary format:
    {
        "SHOULDER": (0.0, 0.0, 0.0),           # Shoulder (origin)
        "ELBOW": (x, y, z),                    # Elbow
        "WRIST": (x, y, z),                    # Wrist (from Pose detection)
        "HAND_WRIST": (x, y, z),               # Wrist (from Hand detection)
        "HAND_THUMB_CMC": (x, y, z),           # Thumb carpometacarpal joint
        "HAND_THUMB_MCP": (x, y, z),           # Thumb metacarpophalangeal joint
        "HAND_THUMB_IP": (x, y, z),            # Thumb interphalangeal joint
        "HAND_THUMB_TIP": (x, y, z),           # Thumb tip
        "HAND_INDEX_FINGER_MCP": (x, y, z),    # Index finger metacarpophalangeal joint
        "HAND_INDEX_FINGER_PIP": (x, y, z),    # Index finger proximal interphalangeal joint
        "HAND_INDEX_FINGER_DIP": (x, y, z),    # Index finger distal interphalangeal joint
        "HAND_INDEX_FINGER_TIP": (x, y, z),    # Index finger tip
        "HAND_MIDDLE_FINGER_MCP": (x, y, z),   # Middle finger metacarpophalangeal joint
        "HAND_MIDDLE_FINGER_PIP": (x, y, z),   # Middle finger proximal interphalangeal joint
        "HAND_MIDDLE_FINGER_DIP": (x, y, z),   # Middle finger distal interphalangeal joint
        "HAND_MIDDLE_FINGER_TIP": (x, y, z),   # Middle finger tip
        "HAND_RING_FINGER_MCP": (x, y, z),     # Ring finger metacarpophalangeal joint
        "HAND_RING_FINGER_PIP": (x, y, z),     # Ring finger proximal interphalangeal joint
        "HAND_RING_FINGER_DIP": (x, y, z),     # Ring finger distal interphalangeal joint
        "HAND_RING_FINGER_TIP": (x, y, z),     # Ring finger tip
        "HAND_PINKY_MCP": (x, y, z),           # Pinky metacarpophalangeal joint
        "HAND_PINKY_PIP": (x, y, z),           # Pinky proximal interphalangeal joint
        "HAND_PINKY_DIP": (x, y, z),           # Pinky distal interphalangeal joint
        "HAND_PINKY_TIP": (x, y, z),           # Pinky tip
    }

Coordinate unit: meters (based on MediaPipe depth estimation)

Note: Returned coordinates have been length-normalized and conform to standard lengths defined in human_config.json.

================================================================================
URDF Standard Coordinate System Definition (with shoulder as origin)
================================================================================

    Z-axis (up)
      |
      |
      |_______ X-axis (forward)
     /
    /
   Y-axis (left)

- X-axis: Points forward (positive is forward)
- Y-axis: Points to the left side of the body (positive is left)
- Z-axis: Points upward (positive is up)

Both left and right arms use the same coordinate axis directions, just different origins (left shoulder and right shoulder respectively).
This conforms to standard robot coordinate system definitions, where all joint rotation axes are Z-axis.

================================================================================
Coordinate Calculation Principle
================================================================================

1. Raw data acquisition
   - Use MediaPipe Pose to detect body keypoints (shoulder, elbow, wrist, hip, etc.)
   - Use MediaPipe Hands to detect 21 hand keypoints
   - PoseHandFusion module fuses both and outputs 3D coordinates in world coordinate system

2. Establish body local coordinate system
   For each side of the arm, establish coordinate system with shoulder as origin:

   a) Determine Y-axis (downward):
      Y = normalize(hip - shoulder)
      Unit vector from shoulder to hip on the same side

   b) Determine reference right direction:
      right_dir = normalize(right_shoulder - left_shoulder)
      Unit vector from left shoulder to right shoulder

   c) Determine Z-axis (forward):
      Z = normalize(cross(right_dir, Y))
      Right-hand rule: right × down = forward

   d) Determine X-axis (rightward):
      X = normalize(cross(Y, Z))
      Right-hand rule: down × forward = right

   e) Build rotation matrix:
      R = [X | Y | Z]  (3x3 matrix, column vectors are each axis)

3. Coordinate transformation
   Transform world coordinate point P_world to local coordinate P_local:

   P_local = R^T @ (P_world - shoulder)

   Where:
   - shoulder is the shoulder position (local coordinate system origin)
   - R^T is the transpose of rotation matrix (equal to inverse matrix)

4. URDF coordinate system transformation
   Transform first-person perspective coordinate system to URDF standard coordinate system:

   [x_urdf, y_urdf, z_urdf] = [z_fp, -x_fp, -y_fp]

   Where:
   - First-person: X(right), Y(down), Z(forward)
   - URDF standard: X(forward), Y(left), Z(up)

================================================================================
Usage Examples
================================================================================

# Method 1: Class interface (recommended for continuous processing)
with ArmPoseAPI(is_selfie=False) as api:
    left_arm, right_arm = api.get_arm_coordinates(frame)

    # Get right hand index finger tip position
    if "HAND_INDEX_FINGER_TIP" in right_arm:
        x, y, z = right_arm["HAND_INDEX_FINGER_TIP"]
        print(f"Right index tip: forward {x:.3f}m, left {y:.3f}m, up {z:.3f}m")

    # Length normalization note:
    # Returned coordinates have been length-normalized and conform to standard lengths defined in human_config.json
    # For example: distance from shoulder to elbow is always 35cm (regardless of actual detected distance)

# Method 2: Function interface (suitable for single frame processing)
left_arm, right_arm = get_arm_pose_from_frame(frame)

# Method 3: Use custom human configuration file
with ArmPoseAPI(human_config_path="custom_human_config.json") as api:
    left_arm, right_arm = api.get_arm_coordinates(frame)

# Method 4: Check length normalization effect
left_arm, right_arm = get_arm_pose_from_frame(frame)

# Calculate actual distance from shoulder to elbow
if "ELBOW" in left_arm:
    import math
    shoulder = left_arm["SHOULDER"]  # (0, 0, 0)
    elbow = left_arm["ELBOW"]
    actual_distance = math.sqrt(
        (elbow[0] - shoulder[0])**2 +
        (elbow[1] - shoulder[1])**2 +
        (elbow[2] - shoulder[2])**2
    )
    print(f"Shoulder to elbow distance: {actual_distance:.3f}m (should be 0.350m)")

================================================================================
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from .fusion_module import PoseHandFusion
from .arm_coordinate_transformer import ArmCoordinateTransformer


class ArmPoseAPI:
    """Arm Pose API, provides URDF standard coordinate system relative coordinates for both arms"""

    def __init__(self, config_path: str = "skeleton_config.json", is_selfie: bool = False, human_config_path: str = "human_config.json"):
        """
        Initialize API

        Args:
            config_path: Skeleton configuration file path
            is_selfie: Whether to use selfie mode (front-facing camera facing yourself)
            human_config_path: Path to human configuration file for length normalization
        """
        self.detector = PoseHandFusion(config_path, is_selfie=is_selfie)
        self.transformer = ArmCoordinateTransformer(use_urdf_standard=True, config_path=human_config_path)

    def get_arm_coordinates(
        self, frame: np.ndarray
    ) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
        """
        Get URDF standard coordinate system relative coordinates for both arms from image frame

        Args:
            frame: RGB format image frame (numpy array)

        Returns:
            (left_arm, right_arm): Two dictionaries, format {joint name: (x, y, z)}

            Joint names include:
            - Arm: "SHOULDER", "ELBOW", "WRIST"
            - Hand: "HAND_WRIST", "HAND_THUMB_CMC", "HAND_THUMB_MCP",
                   "HAND_THUMB_IP", "HAND_THUMB_TIP",
                   "HAND_INDEX_FINGER_MCP", "HAND_INDEX_FINGER_PIP",
                   "HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP",
                   "HAND_MIDDLE_FINGER_MCP", "HAND_MIDDLE_FINGER_PIP",
                   "HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP",
                   "HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP",
                   "HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP",
                   "HAND_PINKY_MCP", "HAND_PINKY_PIP",
                   "HAND_PINKY_DIP", "HAND_PINKY_TIP"

            URDF standard coordinate system (with shoulder as origin):
            - X: Positive value indicates forward direction of body
            - Y: Positive value indicates left direction of body
            - Z: Positive value indicates upward direction

            Returns empty dictionary if detection fails.
        """
        # Detect joints
        joints, _ = self.detector.process_frame(frame)

        # Transform to local coordinate system
        left_local, right_local = self.transformer.compute_local_coordinates(joints)

        # Convert format: list -> tuple
        left_arm = {name: tuple(coords) for name, coords in left_local.items()}
        right_arm = {name: tuple(coords) for name, coords in right_local.items()}

        return left_arm, right_arm

    def stop(self):
        """Stop detector and release resources"""
        self.detector.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def get_arm_pose_from_frame(
    frame: np.ndarray,
    detector: Optional[PoseHandFusion] = None,
    transformer: Optional[ArmCoordinateTransformer] = None,
    config_path: str = "skeleton_config.json",
    is_selfie: bool = False,
    human_config_path: str = "human_config.json"
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
    """
    Convenience function to get URDF standard coordinate system relative coordinates for both arms from a single frame

    Args:
        frame: BGR format image frame
        detector: Optional, reuse existing detector
        transformer: Optional, reuse existing coordinate transformer (default uses URDF standard)
        config_path: Skeleton configuration file path (only used when detector is None)
        is_selfie: Whether to use selfie mode
        human_config_path: Path to human configuration file for length normalization (only used when transformer is None)

    Returns:
        (left_arm, right_arm): Two dictionaries, format {joint name: (x, y, z)}

        URDF standard coordinate system (with shoulder as origin):
        - X: Positive value indicates forward direction of body
        - Y: Positive value indicates left direction of body
        - Z: Positive value indicates upward direction
    """
    # Create or reuse detector
    if detector is None:
        detector = PoseHandFusion(config_path, is_selfie=is_selfie)
        should_stop = True
    else:
        should_stop = False

    if transformer is None:
        transformer = ArmCoordinateTransformer(use_urdf_standard=True, config_path=human_config_path)

    try:
        # Detect joints
        joints, _ = detector.process_frame(frame)

        # Transform to local coordinate system
        left_local, right_local = transformer.compute_local_coordinates(joints)

        # Convert format
        left_arm = {name: tuple(coords) for name, coords in left_local.items()}
        right_arm = {name: tuple(coords) for name, coords in right_local.items()}

        return left_arm, right_arm
    finally:
        if should_stop:
            detector.stop()


# Usage examples
if __name__ == "__main__":
    # Example 1: Use class interface (recommended for continuous processing)
    print("=" * 50)
    print("Arm Pose API Example (URDF Standard Coordinate System)")
    print("=" * 50)

    with ArmPoseAPI(is_selfie=False) as api:
        cap = cv2.VideoCapture(0)

        print("Press 'q' to quit, 's' to print current coordinates")
        print("URDF standard coordinate system: X(forward), Y(left), Z(up)")
        print("=" * 50)

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            left_arm, right_arm = api.get_arm_coordinates(frame)

            # Display frame
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n--- Current Coordinates (URDF Standard) ---")
                print("Left arm:")
                for name, coords in left_arm.items():
                    print(f"  {name}: forward {coords[0]:.4f}m, left {coords[1]:.4f}m, up {coords[2]:.4f}m")
                print("Right arm:")
                for name, coords in right_arm.items():
                    print(f"  {name}: forward {coords[0]:.4f}m, left {coords[1]:.4f}m, up {coords[2]:.4f}m")

        cap.release()
        cv2.destroyAllWindows()
