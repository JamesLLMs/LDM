"""
Arm Coordinate Transformer

Functions:
1. Establish independent coordinate system with left/right shoulder as origin
2. Use shoulder-hip plane as reference plane
3. Transform arm and hand coordinates to shoulder coordinate system
4. Apply length normalization to ensure URDF joint distances are not affected by viewing angle changes

Front camera note:
fusion_module's hand_worker flips hand labels when is_selfie=False,
so LEFT_* in joints_dict corresponds to the user's true left side, RIGHT_* to the right side.
Pose joint labels are based on the detected person's anatomical left/right and do not need flipping.
Therefore, this class directly uses labels in joints_dict without any left-right swapping.

URDF standard coordinate system definition (when use_urdf_standard=True):
- Origin: Shoulder position
- X-axis: Points forward (positive is forward)
- Y-axis: Points to left side of body (positive is left)
- Z-axis: Points upward (positive is up)

First-person perspective coordinate system (when use_urdf_standard=False):
- Origin: Shoulder position
- X-axis: Points to right side of body (positive is right)
- Y-axis: From shoulder to hip (positive is down)
- Z-axis: Points forward (positive is forward)

URDF coordinate system follows standard robotics conventions, all joint rotation axes are Z-axis.

Length normalization:
Uses standard lengths defined in human_config.json to scale distances between joints,
ensuring URDF output length parameters remain fixed and do not change due to viewing angle variations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ArmCoordinateTransformer:
    """Arm coordinate transformer, converts world coordinates to local coordinate system with shoulder as origin"""

    def __init__(self, use_urdf_standard: bool = True, config_path: str = "human_config.json"):
        """
        Initialize coordinate transformer

        Args:
            use_urdf_standard: Whether to use URDF standard coordinate system
                True: Use URDF standard (X forward, Y left, Z up) - robot coordinate system
                False: Use first-person perspective (X right, Y down, Z forward)
            config_path: Path to human configuration file, used to load standard joint lengths
        """
        # Store computed coordinate systems
        self.left_shoulder_frame = None   # (origin, rotation_matrix)
        self.right_shoulder_frame = None

        # Whether to use URDF standard coordinate system
        self.use_urdf_standard = use_urdf_standard

        # Find configuration file (support multiple possible paths)
        config_file = self._find_config_file(config_path)

        # Load human configuration (standard joint lengths)
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.human_config = json.load(f)
                print(f"Loaded human configuration file: {config_file}")
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_path} not found, skipping length normalization")
            self.human_config = {"left_arm": {}, "right_arm": {}}
        except json.JSONDecodeError:
            print(f"Warning: Configuration file {config_path} format error, skipping length normalization")
            self.human_config = {"left_arm": {}, "right_arm": {}}

        # Load coordinate rotation matrix (used to align human coordinate system to robot arm coordinate system)
        self.coordinate_rotation_matrix = None
        if "coordinate_transform" in self.human_config:
            transform_config = self.human_config["coordinate_transform"]
            if "rotation_matrix" in transform_config:
                self.coordinate_rotation_matrix = np.array(transform_config["rotation_matrix"])
                print(f"Loaded coordinate rotation matrix:\n{self.coordinate_rotation_matrix}")
        # URDF standard coordinate system transformation matrix
        # Transform first-person perspective coordinate system to URDF standard coordinate system
        # First-person: X(right), Y(down), Z(forward)
        # URDF standard: X(forward), Y(left), Z(up)
        # Transformation: [x_urdf, y_urdf, z_urdf] = [z_fp, -x_fp, -y_fp]
        self.urdf_transform_matrix = np.array([
            [0, 0, 1],   # X_urdf = Z_fp
            [-1, 0, 0],  # Y_urdf = -X_fp
            [0, -1, 0]   # Z_urdf = -Y_fp
        ])

        # Hand joint names (21 total)
        self.hand_joint_names = [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]

    def _apply_urdf_transform(self, point: np.ndarray) -> np.ndarray:
        """
        应用URDF标准坐标系转换

        Args:
            point: 第一人称视角坐标系中的点

        Returns:
            URDF标准坐标系中的点
        """
        if not self.use_urdf_standard:
            return point
        return self.urdf_transform_matrix @ point

    def _find_config_file(self, config_path: str) -> str:
        """
        查找配置文件，支持多个可能的路径

        Args:
            config_path: 配置文件路径（可以是相对路径或绝对路径）

        Returns:
            找到的配置文件的完整路径
        """
        # 尝试多个可能的路径
        possible_paths = [
            Path(config_path),  # 原始路径
            Path(__file__).parent / config_path,  # 相对于当前模块的路径
            Path(__file__).parent.parent.parent / "src" / "whobody_dect" / config_path,  # 项目根目录
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # 如果都找不到，返回原始路径（让后续代码处理 FileNotFoundError）
        return config_path

    def _apply_coordinate_rotation(self, point: np.ndarray) -> np.ndarray:
        """
        应用坐标系旋转矩阵（将人体坐标系对齐到机械臂坐标系）

        Args:
            point: URDF坐标系中的点

        Returns:
            旋转后的点（对齐到机械臂坐标系）
        """
        if self.coordinate_rotation_matrix is None:
            return point
        return self.coordinate_rotation_matrix @ point

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """归一化向量"""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0])
        return v / norm

    def _calculate_distance(self, p1: List[float], p2: List[float]) -> float:
        """
        计算两点之间的欧氏距离

        Args:
            p1: 点1的坐标 [x, y, z]
            p2: 点2的坐标 [x, y, z]

        Returns:
            两点之间的距离（米）
        """
        if p1 is None or p2 is None:
            return 0.0
        p1_np = np.array(p1)
        p2_np = np.array(p2)
        return np.linalg.norm(p2_np - p1_np)

    def _apply_length_normalization(self,
                                    arm_data: Dict[str, List[float]],
                                    side: str) -> Dict[str, List[float]]:
        """
        对手臂坐标应用长度归一化（链式处理，带位移传递）

        按照骨骼树拓扑从根部开始逐级缩放：
        1. 肩膀固定为原点
        2. 缩放每个连杆时，保持方向不变，只调整长度
        3. 将产生的位移量传递给该连杆的所有下游关节

        骨骼树拓扑：
        SHOULDER → ELBOW → WRIST
            ├── THUMB_CMC → THUMB_MCP → THUMB_IP → THUMB_TIP
            └── INDEX_MCP → INDEX_PIP → INDEX_DIP → INDEX_TIP
                └── MIDDLE_MCP → MIDDLE_PIP → MIDDLE_DIP → MIDDLE_TIP
                    └── RING_MCP → RING_PIP → RING_DIP → RING_TIP
                        └── PINKY_MCP → PINKY_PIP → PINKY_DIP → PINKY_TIP

        Args:
            arm_data: 手臂坐标字典 {关节名称: [x, y, z]}
            side: "left" 或 "right"

        Returns:
            归一化后的手臂坐标字典
        """
        if not arm_data or f"{side}_arm" not in self.human_config:
            return arm_data

        # 将所有坐标转换为numpy数组以便计算
        normalized_data = {}
        for key, value in arm_data.items():
            normalized_data[key] = np.array(value)

        side_config = self.human_config.get(f"{side}_arm", {})

        # 定义归一化链：按照骨骼树拓扑排序
        # 每条记录: (父关节key, 子关节key, config_key, 所有下游关节key列表)
        # 下游关节 = 该子关节在树中的所有后代（当子关节位移时，这些关节也要跟着平移）
        normalization_chain = [
            # SHOULDER → ELBOW（下游：WRIST及所有手指）
            ("SHOULDER", "ELBOW", "SHOULDER_TO_ELBOW",
             ["WRIST", "HAND_WRIST",
              "HAND_THUMB_CMC", "HAND_THUMB_MCP", "HAND_THUMB_IP", "HAND_THUMB_TIP",
              "HAND_INDEX_FINGER_MCP", "HAND_INDEX_FINGER_PIP", "HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP",
              "HAND_MIDDLE_FINGER_MCP", "HAND_MIDDLE_FINGER_PIP", "HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP",
              "HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP",
              "HAND_PINKY_MCP", "HAND_PINKY_PIP", "HAND_PINKY_DIP", "HAND_PINKY_TIP"]),

            # ELBOW → WRIST（下游：所有手指）
            ("ELBOW", "WRIST", "ELBOW_TO_WRIST",
             ["HAND_WRIST",
              "HAND_THUMB_CMC", "HAND_THUMB_MCP", "HAND_THUMB_IP", "HAND_THUMB_TIP",
              "HAND_INDEX_FINGER_MCP", "HAND_INDEX_FINGER_PIP", "HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP",
              "HAND_MIDDLE_FINGER_MCP", "HAND_MIDDLE_FINGER_PIP", "HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP",
              "HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP",
              "HAND_PINKY_MCP", "HAND_PINKY_PIP", "HAND_PINKY_DIP", "HAND_PINKY_TIP"]),

            # --- 拇指链：WRIST → CMC → MCP → IP → TIP ---
            ("WRIST", "HAND_THUMB_CMC", "WRIST_TO_THUMB_CMC",
             ["HAND_THUMB_MCP", "HAND_THUMB_IP", "HAND_THUMB_TIP"]),
            ("HAND_THUMB_CMC", "HAND_THUMB_MCP", "THUMB_CMC_TO_THUMB_MCP",
             ["HAND_THUMB_IP", "HAND_THUMB_TIP"]),
            ("HAND_THUMB_MCP", "HAND_THUMB_IP", "THUMB_MCP_TO_THUMB_IP",
             ["HAND_THUMB_TIP"]),
            ("HAND_THUMB_IP", "HAND_THUMB_TIP", "THUMB_IP_TO_THUMB_TIP",
             []),

            # --- 食指链：WRIST → INDEX_MCP → PIP → DIP → TIP ---
            # INDEX_MCP 的下游包含自身的 PIP/DIP/TIP 和整个 MCP 链后续的手指
            ("WRIST", "HAND_INDEX_FINGER_MCP", "WRIST_TO_INDEX_FINGER_MCP",
             ["HAND_INDEX_FINGER_PIP", "HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP",
              "HAND_MIDDLE_FINGER_MCP", "HAND_MIDDLE_FINGER_PIP", "HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP",
              "HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP",
              "HAND_PINKY_MCP", "HAND_PINKY_PIP", "HAND_PINKY_DIP", "HAND_PINKY_TIP"]),
            ("HAND_INDEX_FINGER_MCP", "HAND_INDEX_FINGER_PIP", "INDEX_FINGER_MCP_TO_INDEX_FINGER_PIP",
             ["HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP"]),
            ("HAND_INDEX_FINGER_PIP", "HAND_INDEX_FINGER_DIP", "INDEX_FINGER_PIP_TO_INDEX_FINGER_DIP",
             ["HAND_INDEX_FINGER_TIP"]),
            ("HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP", "INDEX_FINGER_DIP_TO_INDEX_FINGER_TIP",
             []),

            # --- MCP链 → 中指链：INDEX_MCP → MIDDLE_MCP → PIP → DIP → TIP ---
            ("HAND_INDEX_FINGER_MCP", "HAND_MIDDLE_FINGER_MCP", "INDEX_FINGER_MCP_TO_MIDDLE_FINGER_MCP",
             ["HAND_MIDDLE_FINGER_PIP", "HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP",
              "HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP",
              "HAND_PINKY_MCP", "HAND_PINKY_PIP", "HAND_PINKY_DIP", "HAND_PINKY_TIP"]),
            ("HAND_MIDDLE_FINGER_MCP", "HAND_MIDDLE_FINGER_PIP", "MIDDLE_FINGER_MCP_TO_MIDDLE_FINGER_PIP",
             ["HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP"]),
            ("HAND_MIDDLE_FINGER_PIP", "HAND_MIDDLE_FINGER_DIP", "MIDDLE_FINGER_PIP_TO_MIDDLE_FINGER_DIP",
             ["HAND_MIDDLE_FINGER_TIP"]),
            ("HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP", "MIDDLE_FINGER_DIP_TO_MIDDLE_FINGER_TIP",
             []),

            # --- MCP链 → 无名指链：MIDDLE_MCP → RING_MCP → PIP → DIP → TIP ---
            ("HAND_MIDDLE_FINGER_MCP", "HAND_RING_FINGER_MCP", "MIDDLE_FINGER_MCP_TO_RING_FINGER_MCP",
             ["HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP",
              "HAND_PINKY_MCP", "HAND_PINKY_PIP", "HAND_PINKY_DIP", "HAND_PINKY_TIP"]),
            ("HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP", "RING_FINGER_MCP_TO_RING_FINGER_PIP",
             ["HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP"]),
            ("HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP", "RING_FINGER_PIP_TO_RING_FINGER_DIP",
             ["HAND_RING_FINGER_TIP"]),
            ("HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP", "RING_FINGER_DIP_TO_RING_FINGER_TIP",
             []),

            # --- MCP链 → 小指链：RING_MCP → PINKY_MCP → PIP → DIP → TIP ---
            ("HAND_RING_FINGER_MCP", "HAND_PINKY_MCP", "RING_FINGER_MCP_TO_PINKY_MCP",
             ["HAND_PINKY_PIP", "HAND_PINKY_DIP", "HAND_PINKY_TIP"]),
            ("HAND_PINKY_MCP", "HAND_PINKY_PIP", "PINKY_MCP_TO_PINKY_PIP",
             ["HAND_PINKY_DIP", "HAND_PINKY_TIP"]),
            ("HAND_PINKY_PIP", "HAND_PINKY_DIP", "PINKY_PIP_TO_PINKY_DIP",
             ["HAND_PINKY_TIP"]),
            ("HAND_PINKY_DIP", "HAND_PINKY_TIP", "PINKY_DIP_TO_PINKY_TIP",
             []),
        ]

        # 按照拓扑顺序依次处理每个连杆
        for parent_key, child_key, config_key, downstream_keys in normalization_chain:
            if parent_key not in normalized_data or child_key not in normalized_data:
                continue

            target_length = side_config.get(config_key)
            if target_length is None:
                continue

            parent_pos = normalized_data[parent_key]
            child_pos = normalized_data[child_key]

            direction = child_pos - parent_pos
            current_length = np.linalg.norm(direction)

            if current_length > 1e-6:
                # 保持原有方向，仅调整长度
                new_child_pos = parent_pos + direction * (target_length / current_length)

                # 计算位移量
                displacement = new_child_pos - child_pos

                # 将位移传递给所有下游关节
                for downstream_key in downstream_keys:
                    if downstream_key in normalized_data:
                        normalized_data[downstream_key] = normalized_data[downstream_key] + displacement

                # 更新当前关节位置
                normalized_data[child_key] = new_child_pos

        # WRIST 和 HAND_WRIST 保持同步
        if "WRIST" in normalized_data:
            normalized_data["HAND_WRIST"] = normalized_data["WRIST"].copy()

        # 将numpy数组转回list
        result = {}
        for key, value in normalized_data.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value

        return result

    def _build_coordinate_frame(self,
                                 shoulder_pos: np.ndarray,
                                 hip_pos: np.ndarray,
                                 left_shoulder: np.ndarray,
                                 right_shoulder: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        建立以肩膀为原点的坐标系。
        两侧使用完全相同的轴方向定义，只是原点不同。

        Args:
            shoulder_pos: 当前肩膀位置（作为原点）
            hip_pos: 对应侧的胯部位置（用于确定Y轴方向）
            left_shoulder: 左肩位置（用于确定平面）
            right_shoulder: 右肩位置（用于确定平面）

        Returns:
            (origin, rotation_matrix): 原点和旋转矩阵（列向量为X,Y,Z轴）
        """
        origin = shoulder_pos.copy()

        # Y轴：从肩膀指向胯（向下为正）
        y_axis = self._normalize(hip_pos - shoulder_pos)

        # 参考右方向：从左肩指向右肩
        right_dir = self._normalize(right_shoulder - left_shoulder)

        # Z轴：指向身体前方
        # cross(right, down) = forward（右手定则）
        z_axis = self._normalize(np.cross(right_dir, y_axis))

        # X轴：指向身体右侧
        # cross(down, forward) = right
        x_axis = self._normalize(np.cross(y_axis, z_axis))

        # 旋转矩阵：列向量为 X, Y, Z 轴
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        return origin, rotation_matrix

    def _transform_to_local(self,
                            point: np.ndarray,
                            origin: np.ndarray,
                            rotation_matrix: np.ndarray) -> np.ndarray:
        """将世界坐标点转换到局部坐标系"""
        translated = point - origin
        local_point = rotation_matrix.T @ translated
        return local_point

    def compute_local_coordinates(self,
                                   joints_dict: Dict[str, List[float]]
                                   ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        计算左右手臂在各自肩膀坐标系下的局部坐标。

        joints_dict 中的标签已经是正确的解剖学标签：
        - LEFT_* = 用户的左侧
        - RIGHT_* = 用户的右侧

        Args:
            joints_dict: 关节字典 {关节名称: [x, y, z]}

        Returns:
            (left_arm_local, right_arm_local): 左右手臂的局部坐标字典
        """
        left_arm_local = {}
        right_arm_local = {}

        # 检查必要关节
        required = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        for joint in required:
            if joint not in joints_dict:
                return left_arm_local, right_arm_local

        left_shoulder = np.array(joints_dict["LEFT_SHOULDER"])
        right_shoulder = np.array(joints_dict["RIGHT_SHOULDER"])
        left_hip = np.array(joints_dict["LEFT_HIP"])
        right_hip = np.array(joints_dict["RIGHT_HIP"])

        # 建立左肩坐标系
        left_origin, left_rot = self._build_coordinate_frame(
            left_shoulder, left_hip, left_shoulder, right_shoulder
        )
        self.left_shoulder_frame = (left_origin, left_rot)

        # 建立右肩坐标系（方向一致，原点不同）
        right_origin, right_rot = self._build_coordinate_frame(
            right_shoulder, right_hip, left_shoulder, right_shoulder
        )
        self.right_shoulder_frame = (right_origin, right_rot)

        # --- 转换左臂关节 ---
        left_arm_local["SHOULDER"] = [0.0, 0.0, 0.0]

        if "LEFT_ELBOW" in joints_dict:
            p = np.array(joints_dict["LEFT_ELBOW"])
            local_p = self._transform_to_local(p, left_origin, left_rot)
            left_arm_local["ELBOW"] = self._apply_urdf_transform(local_p).tolist()

        # 只使用融合后的Hand手腕，永久移除身体的WRIST
        if "LEFT_HAND_WRIST" in joints_dict:
            p = np.array(joints_dict["LEFT_HAND_WRIST"])
            local_p = self._transform_to_local(p, left_origin, left_rot)
            left_arm_local["WRIST"] = self._apply_urdf_transform(local_p).tolist()
            left_arm_local["HAND_WRIST"] = self._apply_urdf_transform(local_p).tolist()

        for jn in self.hand_joint_names:
            key = f"LEFT_HAND_{jn}"
            if key in joints_dict:
                p = np.array(joints_dict[key])
                local_p = self._transform_to_local(p, left_origin, left_rot)
                left_arm_local[f"HAND_{jn}"] = self._apply_urdf_transform(local_p).tolist()

        # --- 转换右臂关节 ---
        right_arm_local["SHOULDER"] = [0.0, 0.0, 0.0]

        if "RIGHT_ELBOW" in joints_dict:
            p = np.array(joints_dict["RIGHT_ELBOW"])
            local_p = self._transform_to_local(p, right_origin, right_rot)
            right_arm_local["ELBOW"] = self._apply_urdf_transform(local_p).tolist()

        # 只使用融合后的Hand手腕，永久移除身体的WRIST
        if "RIGHT_HAND_WRIST" in joints_dict:
            p = np.array(joints_dict["RIGHT_HAND_WRIST"])
            local_p = self._transform_to_local(p, right_origin, right_rot)
            right_arm_local["WRIST"] = self._apply_urdf_transform(local_p).tolist()
            right_arm_local["HAND_WRIST"] = self._apply_urdf_transform(local_p).tolist()

        for jn in self.hand_joint_names:
            key = f"RIGHT_HAND_{jn}"
            if key in joints_dict:
                p = np.array(joints_dict[key])
                local_p = self._transform_to_local(p, right_origin, right_rot)
                right_arm_local[f"HAND_{jn}"] = self._apply_urdf_transform(local_p).tolist()

        # --- 应用长度归一化 ---
        left_arm_normalized = self._apply_length_normalization(left_arm_local, "left")
        right_arm_normalized = self._apply_length_normalization(right_arm_local, "right")

        # --- 应用坐标系旋转（对齐到机械臂坐标系）---
        left_arm_rotated = self._apply_coordinate_rotation_to_dict(left_arm_normalized)
        right_arm_rotated = self._apply_coordinate_rotation_to_dict(right_arm_normalized)

        return left_arm_rotated, right_arm_rotated

    def _apply_coordinate_rotation_to_dict(self, arm_data: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        对整个关节字典应用坐标系旋转

        Args:
            arm_data: 关节坐标字典 {关节名称: [x, y, z]}

        Returns:
            旋转后的关节坐标字典
        """
        if self.coordinate_rotation_matrix is None:
            return arm_data

        rotated_data = {}
        for key, value in arm_data.items():
            point = np.array(value)
            rotated_point = self._apply_coordinate_rotation(point)
            rotated_data[key] = rotated_point.tolist()

        return rotated_data

    def get_arm_connections(self) -> List[List[str]]:
        """获取手臂骨架连接关系"""
        return [
            ["SHOULDER", "ELBOW"],
            ["ELBOW", "WRIST"],
            ["WRIST", "HAND_WRIST"],
            ["HAND_WRIST", "HAND_THUMB_CMC"],
            ["HAND_THUMB_CMC", "HAND_THUMB_MCP"],
            ["HAND_THUMB_MCP", "HAND_THUMB_IP"],
            ["HAND_THUMB_IP", "HAND_THUMB_TIP"],
            ["HAND_WRIST", "HAND_INDEX_FINGER_MCP"],
            ["HAND_INDEX_FINGER_MCP", "HAND_INDEX_FINGER_PIP"],
            ["HAND_INDEX_FINGER_PIP", "HAND_INDEX_FINGER_DIP"],
            ["HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP"],
            ["HAND_INDEX_FINGER_MCP", "HAND_MIDDLE_FINGER_MCP"],
            ["HAND_MIDDLE_FINGER_MCP", "HAND_MIDDLE_FINGER_PIP"],
            ["HAND_MIDDLE_FINGER_PIP", "HAND_MIDDLE_FINGER_DIP"],
            ["HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP"],
            ["HAND_MIDDLE_FINGER_MCP", "HAND_RING_FINGER_MCP"],
            ["HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP"],
            ["HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP"],
            ["HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP"],
            ["HAND_RING_FINGER_MCP", "HAND_PINKY_MCP"],
            ["HAND_WRIST", "HAND_PINKY_MCP"],
            ["HAND_PINKY_MCP", "HAND_PINKY_PIP"],
            ["HAND_PINKY_PIP", "HAND_PINKY_DIP"],
            ["HAND_PINKY_DIP", "HAND_PINKY_TIP"],
        ]

    def get_coordinate_axes(self, side: str = "left") -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """获取坐标系的轴向量（用于可视化）"""
        frame = self.left_shoulder_frame if side == "left" else self.right_shoulder_frame
        if frame is None:
            return None
        origin, rotation_matrix = frame
        return origin, rotation_matrix[:, 0], rotation_matrix[:, 1], rotation_matrix[:, 2]
