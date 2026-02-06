"""
Whole-body detector wrapper

Converts ArmPoseAPI output to a format compatible with the existing retargeting framework.

Keypoint index definition:
    0: SHOULDER      - Shoulder (origin)
    1: ELBOW         - Elbow
    2: WRIST         - Wrist (from Pose detection)
    3: HAND_WRIST    - Wrist (from Hand detection)
    4-7: THUMB       - Thumb (CMC, MCP, IP, TIP)
    8-11: INDEX      - Index finger (MCP, PIP, DIP, TIP)
    12-15: MIDDLE   - Middle finger (MCP, PIP, DIP, TIP)
    16-19: RING     - Ring finger (MCP, PIP, DIP, TIP)
    20-23: PINKY    - Pinky finger (MCP, PIP, DIP, TIP)

Coordinate system: URDF standard (X forward, Y left, Z up), with shoulder as origin
"""

import numpy as np
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

# Add detection module path
# _current_dir = Path(__file__).parent.absolute()
# _project_root = _current_dir.parent.parent
# _whobody_dect_path = _project_root / "src" / "whobody_dect"
# sys.path.insert(0, str(_whobody_dect_path))

from whobody_dect.arm_pose_api import ArmPoseAPI


class WholeBodyDetector:
    """
    Whole-body detector, wraps ArmPoseAPI to adapt to the existing retargeting framework.

    Output format is compatible with SingleHandDetector, but the number of keypoints is extended from 21 to 24.
    """

    # Keypoint name to index mapping (24 keypoints total)
    KEYPOINT_NAMES = [
        # Arm keypoints (0-2)
        "SHOULDER",
        "ELBOW",
        "WRIST",
        # Hand keypoints (3-23)
        "HAND_WRIST",
        "HAND_THUMB_CMC",
        "HAND_THUMB_MCP",
        "HAND_THUMB_IP",
        "HAND_THUMB_TIP",
        "HAND_INDEX_FINGER_MCP",
        "HAND_INDEX_FINGER_PIP",
        "HAND_INDEX_FINGER_DIP",
        "HAND_INDEX_FINGER_TIP",
        "HAND_MIDDLE_FINGER_MCP",
        "HAND_MIDDLE_FINGER_PIP",
        "HAND_MIDDLE_FINGER_DIP",
        "HAND_MIDDLE_FINGER_TIP",
        "HAND_RING_FINGER_MCP",
        "HAND_RING_FINGER_PIP",
        "HAND_RING_FINGER_DIP",
        "HAND_RING_FINGER_TIP",
        "HAND_PINKY_MCP",
        "HAND_PINKY_PIP",
        "HAND_PINKY_DIP",
        "HAND_PINKY_TIP",
    ]

    # Number of keypoints
    NUM_KEYPOINTS = 24

    def __init__(
        self,
        hand_type: str = "Left",
        is_selfie: bool = False,
        config_path: str = "skeleton_config.json",
        human_config_path: str = "human_config.json",
    ):
        """
        Initialize the detector

        Args:
            hand_type: "Left" or "Right"
            is_selfie: Whether to use selfie mode
            config_path: Skeleton configuration file path
            human_config_path: Human configuration file path
        """
        self.hand_type = hand_type.lower()
        self.is_selfie = is_selfie

        # Initialize ArmPoseAPI
        self.api = ArmPoseAPI(
            config_path=config_path,
            is_selfie=is_selfie,
            human_config_path=human_config_path,
        )

    def detect(
        self, rgb: np.ndarray
    ) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], None]:
        """
        Detect arm and hand keypoints

        Args:
            rgb: RGB format image frame

        Returns:
            (num_detected, joint_pos, keypoint_2d, wrist_rot)
            - num_detected: Number detected (0 or 1)
            - joint_pos: Keypoint 3D coordinate array, shape (24, 3), with shoulder as origin
            - keypoint_2d: None (2D keypoints not currently supported)
            - wrist_rot: None (no coordinate system estimation needed)
        """
        # Call ArmPoseAPI to get coordinates
        left_arm, right_arm = self.api.get_arm_coordinates(rgb)

        # Select corresponding arm data based on hand_type
        arm_data = left_arm if self.hand_type == "left" else right_arm

        # Check if detection succeeded (need at least elbow detected)
        if not arm_data or "ELBOW" not in arm_data:
            return 0, None, None, None

        # Convert dictionary to numpy array
        joint_pos = self._dict_to_array(arm_data)

        return 1, joint_pos, None, None

    def _dict_to_array(
        self, arm_data: Dict[str, Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Convert dictionary returned by ArmPoseAPI to numpy array

        Args:
            arm_data: Joint coordinate dictionary

        Returns:
            Numpy array with shape (24, 3)
        """
        joint_pos = np.zeros((self.NUM_KEYPOINTS, 3), dtype=np.float32)

        for idx, name in enumerate(self.KEYPOINT_NAMES):
            if name in arm_data:
                joint_pos[idx] = arm_data[name]
            elif name == "WRIST" and "HAND_WRIST" in arm_data:
                # If no WRIST but have HAND_WRIST, use HAND_WRIST
                joint_pos[idx] = arm_data["HAND_WRIST"]

        return joint_pos

    def stop(self):
        """Stop the detector and release resources"""
        self.api.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# Convenience function
def create_wholebody_detector(
    hand_type: str = "Left", is_selfie: bool = False
) -> WholeBodyDetector:
    """
    Convenience function to create a whole-body detector

    Args:
        hand_type: "Left" or "Right"
        is_selfie: Whether to use selfie mode

    Returns:
        WholeBodyDetector instance
    """
    return WholeBodyDetector(hand_type=hand_type, is_selfie=is_selfie)


if __name__ == "__main__":
    """Test the detector"""
    import cv2

    print("=" * 50)
    print("Whole-body Detector Test")
    print("=" * 50)

    with WholeBodyDetector(hand_type="Left", is_selfie=False) as detector:
        cap = cv2.VideoCapture(0)

        print("Press 'q' to quit, 's' to print current coordinates")
        print("=" * 50)

        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            num_detected, joint_pos, _, _ = detector.detect(rgb)

            # Display frame
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s") and joint_pos is not None:
                print("\n--- Current Keypoint Coordinates ---")
                for idx, name in enumerate(detector.KEYPOINT_NAMES):
                    pos = joint_pos[idx]
                    print(f"  {idx:2d}. {name:30s}: ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f})")

        cap.release()
        cv2.destroyAllWindows()
