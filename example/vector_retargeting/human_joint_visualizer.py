"""
Human joint visualizer module

Displays human joint points (spheres) in SAPIEN scene, can be displayed together with robot model.
Designed as a reusable module, decoupled from the main program.

Usage:
    # Add joint visualization to existing SAPIEN scene
    visualizer = HumanJointVisualizer(scene)
    visualizer.create_joint_spheres()

    # Update in main loop
    visualizer.update_from_array(joint_pos)  # joint_pos: (24, 3) numpy array

    # Load rotation matrix from config file
    rotation_matrix = load_rotation_matrix_from_config("human_config.json")
    visualizer.set_rotation_matrix(rotation_matrix)
"""

import json
import numpy as np
import sapien
from pathlib import Path
from typing import Dict, List, Optional


# Keypoint names output by WholeBodyDetector (24 total)
# Note: Must be consistent with WholeBodyDetector.KEYPOINT_NAMES
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

# Colors for different types of joints (RGBA)
JOINT_COLORS = {
    "SHOULDER": [1.0, 0.0, 0.0, 1.0],    # Red
    "ELBOW": [0.0, 1.0, 0.0, 1.0],       # Green
    "WRIST": [0.0, 0.0, 1.0, 1.0],       # Blue
    "THUMB": [1.0, 0.5, 0.0, 1.0],       # Orange
    "INDEX": [1.0, 1.0, 0.0, 1.0],       # Yellow
    "MIDDLE": [0.0, 1.0, 1.0, 1.0],      # Cyan
    "RING": [1.0, 0.0, 1.0, 1.0],        # Magenta
    "PINKY": [0.5, 0.0, 1.0, 1.0],       # Purple
}


def get_joint_color(joint_name: str) -> List[float]:
    """Get color based on joint name"""
    for key, color in JOINT_COLORS.items():
        if key in joint_name:
            return color
    return [0.5, 0.5, 0.5, 1.0]  # Default gray


class HumanJointVisualizer:
    """
    Human joint visualizer

    Creates spheres in an existing SAPIEN scene to represent human joint positions.
    Supports real-time joint position updates.

    Coordinate system notes:
    - Human detection output coordinate system: URDF standard (X forward, Y left, Z up), with shoulder as origin
    - Can use rotation_matrix to rotate human coordinate system to align with robot coordinate system
    """

    def __init__(
        self,
        scene: sapien.Scene,
        sphere_radius: float = 0.012,
        position_offset: Optional[np.ndarray] = None,
        rotation_matrix: Optional[np.ndarray] = None,
        rotation_quaternion: Optional[List[float]] = None,
        visible: bool = True,
    ):
        """
        Initialize the visualizer

        Args:
            scene: SAPIEN scene object (already created)
            sphere_radius: Joint sphere radius
            position_offset: Position offset [x, y, z], applied after rotation
            rotation_matrix: 3x3 rotation matrix to rotate human coordinate system to target coordinate system
                            Example: Rotate 180° around Z-axis to match UR5e's base_link_inertia
            rotation_quaternion: Quaternion [x, y, z, w], used to set sphere pose rotation
            visible: Whether to be visible by default
        """
        self.scene = scene
        self.sphere_radius = sphere_radius
        self.position_offset = position_offset if position_offset is not None else np.array([0.0, 0.0, 0.0])
        self.rotation_matrix = rotation_matrix  # Can be None, meaning no rotation
        self.rotation_quaternion = rotation_quaternion  # Used for sphere pose
        self.visible = visible

        self.joint_spheres: Dict[str, sapien.Entity] = {}
        self.joint_positions: Dict[str, np.ndarray] = {}
        self._created = False

    def _transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Apply coordinate transformation to a point (rotation first, then translation)

        Args:
            point: Original coordinates [x, y, z]

        Returns:
            Transformed coordinates
        """
        transformed = point.copy()

        # Apply rotation first
        if self.rotation_matrix is not None:
            transformed = self.rotation_matrix @ transformed

        # Then apply translation
        transformed = transformed + self.position_offset

        return transformed

    def create_joint_spheres(self) -> None:
        """Create spheres for all joints"""
        if self._created:
            return

        # Calculate initial pose quaternion (for SAPIEN)
        initial_quat = [1, 0, 0, 0]  # Default no rotation [w, x, y, z]
        if self.rotation_quaternion is not None:
            # Convert format: we store [x, y, z, w], SAPIEN needs [w, x, y, z]
            x, y, z, w = self.rotation_quaternion
            initial_quat = [w, x, y, z]

        for joint_name in KEYPOINT_NAMES:
            color = get_joint_color(joint_name)

            # Create material
            material = sapien.render.RenderMaterial()
            material.base_color = color

            # Create sphere
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=self.sphere_radius,
                material=material
            )
            sphere = builder.build_static(name=f"human_{joint_name}")

            # Initial position at invisible location, but with correct pose
            sphere.set_pose(sapien.Pose(p=[0, 0, -10], q=initial_quat))

            self.joint_spheres[joint_name] = sphere
            self.joint_positions[joint_name] = np.array([0, 0, -10])

        self._created = True

    def update_from_array(self, joint_pos: np.ndarray) -> None:
        """
        Update joint positions from numpy array

        Args:
            joint_pos: Joint position array with shape (24, 3), compatible with WholeBodyDetector output
                       Note: Undetected joint coordinates are [0, 0, 0] and will be hidden
        """
        if joint_pos is None or not self._created:
            return

        if not self.visible:
            return

        for idx, joint_name in enumerate(KEYPOINT_NAMES):
            if idx < len(joint_pos):
                raw_pos = joint_pos[idx]
                # Check if valid coordinates (non-zero coordinates, or SHOULDER origin)
                # SHOULDER coordinates are always [0, 0, 0] (origin), need special handling
                is_valid = (idx == 0) or (np.linalg.norm(raw_pos) > 1e-6)

                if is_valid:
                    # Apply coordinate transformation (rotation + translation)
                    pos = self._transform_point(raw_pos)
                    self._update_sphere(joint_name, pos)
                else:
                    # Undetected joints, hide at invisible location
                    self._hide_sphere(joint_name)

    def _hide_sphere(self, joint_name: str) -> None:
        """Hide a single sphere"""
        if joint_name in self.joint_spheres:
            sphere = self.joint_spheres[joint_name]
            # Keep pose, just move to invisible position
            if self.rotation_quaternion is not None:
                x, y, z, w = self.rotation_quaternion
                sphere.set_pose(sapien.Pose(p=[0, 0, -10], q=[w, x, y, z]))
            else:
                sphere.set_pose(sapien.Pose(p=[0, 0, -10]))
            self.joint_positions[joint_name] = np.array([0, 0, -10])

    def update_from_dict(self, positions: Dict[str, np.ndarray]) -> None:
        """
        Update joint positions from dictionary

        Args:
            positions: {joint_name: [x, y, z]} dictionary
        """
        if not self._created or not self.visible:
            return

        for joint_name, pos in positions.items():
            if joint_name in self.joint_spheres:
                # Apply coordinate transformation (rotation + translation)
                new_pos = self._transform_point(np.array(pos))
                self._update_sphere(joint_name, new_pos)

    def set_rotation_matrix(self, rotation_matrix: Optional[np.ndarray]) -> None:
        """Set rotation matrix"""
        self.rotation_matrix = rotation_matrix

    def _update_sphere(self, joint_name: str, pos: np.ndarray) -> None:
        """Update a single sphere position and pose"""
        if joint_name in self.joint_spheres:
            sphere = self.joint_spheres[joint_name]
            # Set position and rotation (apply if quaternion is available)
            if self.rotation_quaternion is not None:
                # SAPIEN Pose quaternion format is [w, x, y, z]
                # We store [x, y, z, w], need to convert
                x, y, z, w = self.rotation_quaternion
                sphere.set_pose(sapien.Pose(p=pos.tolist(), q=[w, x, y, z]))
            else:
                sphere.set_pose(sapien.Pose(p=pos.tolist()))
            self.joint_positions[joint_name] = pos

    def set_visible(self, visible: bool) -> None:
        """Set visibility"""
        self.visible = visible
        if not visible:
            # Hide all spheres, keep pose
            quat = [1, 0, 0, 0]
            if self.rotation_quaternion is not None:
                x, y, z, w = self.rotation_quaternion
                quat = [w, x, y, z]
            for sphere in self.joint_spheres.values():
                sphere.set_pose(sapien.Pose(p=[0, 0, -10], q=quat))

    def set_position_offset(self, offset: np.ndarray) -> None:
        """Set position offset"""
        self.position_offset = offset

    def get_joint_position(self, joint_name: str) -> Optional[np.ndarray]:
        """Get current position of specified joint"""
        return self.joint_positions.get(joint_name)

    def hide_all(self) -> None:
        """Hide all joint spheres"""
        for sphere in self.joint_spheres.values():
            sphere.set_pose(sapien.Pose(p=[0, 0, -10]))

    @property
    def num_joints(self) -> int:
        """Return number of joints"""
        return len(KEYPOINT_NAMES)

    @staticmethod
    def get_keypoint_names() -> List[str]:
        """Return keypoint names list"""
        return KEYPOINT_NAMES.copy()


def create_rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """
    Create rotation matrix around Z-axis

    Args:
        angle_rad: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def create_rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """
    Create rotation matrix around Y-axis

    Args:
        angle_rad: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def create_rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """
    Create rotation matrix around X-axis

    Args:
        angle_rad: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


# Predefined rotation matrices
ROTATION_Z_180 = create_rotation_matrix_z(np.pi)  # Rotate 180° around Z-axis, for UR5e
ROTATION_IDENTITY = np.eye(3)  # Identity matrix, no rotation


def load_rotation_matrix_from_config(config_path: str) -> Optional[np.ndarray]:
    """
    Load coordinate system rotation matrix from human configuration file

    Args:
        config_path: Configuration file path (can be relative or absolute)

    Returns:
        3x3 rotation matrix, returns None if loading fails
    """
    result = load_coordinate_transform_from_config(config_path)
    if result is None:
        return None
    return result[0]


def load_coordinate_transform_from_config(config_path: str) -> Optional[tuple]:
    """
    Load coordinate system transformation (rotation matrix and quaternion) from human configuration file

    Args:
        config_path: Configuration file path (can be relative or absolute)

    Returns:
        (rotation_matrix, rotation_quaternion) tuple, returns None if loading fails
        - rotation_matrix: 3x3 rotation matrix
        - rotation_quaternion: Quaternion [x, y, z, w]
    """
    # Try multiple possible paths
    possible_paths = [
        Path(config_path),
        Path(__file__).parent / config_path,
        Path(__file__).parent.parent.parent / "src" / "whobody_dect" / config_path,
    ]

    config_file = None
    for path in possible_paths:
        if path.exists():
            config_file = path
            break

    if config_file is None:
        print(f"Warning: Configuration file {config_path} not found")
        return None

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if "coordinate_transform" not in config:
            print(f"Warning: coordinate_transform field not found in configuration file")
            return None

        transform_config = config["coordinate_transform"]

        # Load rotation matrix
        rotation_matrix = None
        if "rotation_matrix" in transform_config:
            rotation_matrix = np.array(transform_config["rotation_matrix"])
            if rotation_matrix.shape != (3, 3):
                print(f"Warning: rotation_matrix shape incorrect, expected (3, 3), got {rotation_matrix.shape}")
                rotation_matrix = None

        # Load quaternion
        rotation_quaternion = None
        if "rotation_quaternion" in transform_config:
            rotation_quaternion = transform_config["rotation_quaternion"]
            if len(rotation_quaternion) != 4:
                print(f"Warning: rotation_quaternion length incorrect, expected 4, got {len(rotation_quaternion)}")
                rotation_quaternion = None

        if rotation_matrix is None and rotation_quaternion is None:
            return None

        return (rotation_matrix, rotation_quaternion)

    except json.JSONDecodeError as e:
        print(f"Warning: Configuration file {config_file} JSON format error: {e}")
        return None
    except Exception as e:
        print(f"Warning: Failed to load configuration file: {e}")
        return None


def quaternion_to_rotation_matrix(quat: List[float]) -> np.ndarray:
    """
    Convert quaternion to rotation matrix

    Args:
        quat: Quaternion [x, y, z, w]

    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
