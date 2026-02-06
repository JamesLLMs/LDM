#!/usr/bin/env python3
"""
Sapien 人体关节可视化器

使用 Sapien 3.x 创建圆球代表人体关节位置，实时接收摄像头检测的姿态数据更新位置。
每个圆球上方显示关节名称标签。
"""

import cv2

try:
    import sapien
    from sapien.utils.viewer import Viewer
except ImportError:
    print("错误：未安装 sapien，请运行: pip install sapien")
    exit(1)

from fusion_module import PoseHandFusion
from arm_coordinate_transformer import ArmCoordinateTransformer


# 所有需要可视化的关节名称
JOINT_NAMES = [
    "SHOULDER",
    "ELBOW",
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

# ArmCoordinateTransformer 输出的关节名到可视化关节名的映射
TRANSFORMER_TO_JOINT = {
    "SHOULDER": "SHOULDER",
    "ELBOW": "ELBOW",
    "WRIST": "WRIST",
    "HAND_WRIST": "WRIST",
    "HAND_THUMB_CMC": "THUMB_CMC",
    "HAND_THUMB_MCP": "THUMB_MCP",
    "HAND_THUMB_IP": "THUMB_IP",
    "HAND_THUMB_TIP": "THUMB_TIP",
    "HAND_INDEX_FINGER_MCP": "INDEX_FINGER_MCP",
    "HAND_INDEX_FINGER_PIP": "INDEX_FINGER_PIP",
    "HAND_INDEX_FINGER_DIP": "INDEX_FINGER_DIP",
    "HAND_INDEX_FINGER_TIP": "INDEX_FINGER_TIP",
    "HAND_MIDDLE_FINGER_MCP": "MIDDLE_FINGER_MCP",
    "HAND_MIDDLE_FINGER_PIP": "MIDDLE_FINGER_PIP",
    "HAND_MIDDLE_FINGER_DIP": "MIDDLE_FINGER_DIP",
    "HAND_MIDDLE_FINGER_TIP": "MIDDLE_FINGER_TIP",
    "HAND_RING_FINGER_MCP": "RING_FINGER_MCP",
    "HAND_RING_FINGER_PIP": "RING_FINGER_PIP",
    "HAND_RING_FINGER_DIP": "RING_FINGER_DIP",
    "HAND_RING_FINGER_TIP": "RING_FINGER_TIP",
    "HAND_PINKY_MCP": "PINKY_MCP",
    "HAND_PINKY_PIP": "PINKY_PIP",
    "HAND_PINKY_DIP": "PINKY_DIP",
    "HAND_PINKY_TIP": "PINKY_TIP",
}

# 不同类型关节的颜色 (RGBA)
JOINT_COLORS = {
    "SHOULDER": [1.0, 0.0, 0.0, 1.0],    # 红色
    "ELBOW": [0.0, 1.0, 0.0, 1.0],       # 绿色
    "WRIST": [0.0, 0.0, 1.0, 1.0],       # 蓝色
    "THUMB": [1.0, 0.5, 0.0, 1.0],       # 橙色
    "INDEX": [1.0, 1.0, 0.0, 1.0],       # 黄色
    "MIDDLE": [0.0, 1.0, 1.0, 1.0],      # 青色
    "RING": [1.0, 0.0, 1.0, 1.0],        # 品红
    "PINKY": [0.5, 0.0, 1.0, 1.0],       # 紫色
}


def get_joint_color(joint_name: str) -> list:
    """根据关节名称获取颜色"""
    for key, color in JOINT_COLORS.items():
        if key in joint_name:
            return color
    return [0.5, 0.5, 0.5, 1.0]  # 默认灰色


class JointVisualizer:
    """人体关节圆球可视化器"""

    def __init__(self, sphere_radius: float = 0.015):
        """
        初始化可视化器

        Args:
            sphere_radius: 圆球半径
        """
        self.scene = None
        self.viewer = None
        self.joint_spheres = {}  # 关节名 -> sphere entity
        self.joint_positions = {}  # 关节名 -> [x, y, z] 当前位置
        self.sphere_radius = sphere_radius
        self.base_height = 0.5  # 基础高度偏移

    def setup_scene(self):
        """设置 Sapien 场景"""
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)
        self.scene.add_ground(altitude=0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light(
            direction=[1, -1, -1], color=[1, 1, 1], shadow=True
        )
        self.scene.add_point_light(position=[0, 0, 2], color=[1, 1, 1])

    def create_joint_spheres(self):
        """预先创建所有关节的圆球"""
        for joint_name in JOINT_NAMES:
            color = get_joint_color(joint_name)

            # 创建材质
            material = sapien.render.RenderMaterial()
            material.base_color = color

            # 创建球体
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=self.sphere_radius,
                material=material
            )
            sphere = builder.build_static(name=joint_name)

            # 初始位置放在地面下（不可见）
            sphere.set_pose(sapien.Pose(p=[0, 0, -1]))

            self.joint_spheres[joint_name] = sphere
            self.joint_positions[joint_name] = [0, 0, -1]

        print(f"创建了 {len(self.joint_spheres)} 个关节圆球")

    def create_viewer(self):
        """创建可视化窗口"""
        self.viewer = Viewer()
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_pose(
            sapien.Pose(p=[1.5, 0, 1.0], q=[0.9239, 0, 0.3827, 0])
        )

    def update_joint_positions(self, positions: dict):
        """
        更新关节位置

        Args:
            positions: {关节名: [x, y, z]}
        """
        for joint_name, pos in positions.items():
            if joint_name in self.joint_spheres:
                sphere = self.joint_spheres[joint_name]
                # 加上基础偏移，使手臂显示在地面上方
                new_pos = [pos[0], pos[1], pos[2] + self.base_height]
                sphere.set_pose(sapien.Pose(p=new_pos))
                self.joint_positions[joint_name] = new_pos

    def draw_color_legend(self, frame):
        """
        在 OpenCV 图像上绘制颜色图例

        Args:
            frame: OpenCV 图像
        """
        y_start = 60
        for name, color in JOINT_COLORS.items():
            # BGR 顺序 (OpenCV)
            bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            # 绘制颜色方块
            cv2.rectangle(frame, (10, y_start), (30, y_start + 15), bgr, -1)
            # 绘制文字
            cv2.putText(frame, name, (35, y_start + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_start += 20

    def run_with_detection(self, side: str = "left"):
        """
        实时检测模式

        Args:
            side: "left" 或 "right"
        """
        detector = PoseHandFusion("skeleton_config.json", is_selfie=False)
        transformer = ArmCoordinateTransformer(use_urdf_standard=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return

        print("\n=== 实时关节可视化模式 ===")
        print(f"使用手臂: {side}")
        print("按 'q' (OpenCV窗口) 或关闭 Sapien 窗口退出")
        print("=" * 40)
        print("\n关节颜色说明:")
        for name, color in JOINT_COLORS.items():
            r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            print(f"  {name}: RGB({r}, {g}, {b})")

        try:
            while not self.viewer.closed:
                success, frame = cap.read()
                if not success:
                    continue

                joints_dict, _ = detector.process_frame(frame)

                if joints_dict:
                    left_arm, right_arm = transformer.compute_local_coordinates(joints_dict)
                    arm_data = left_arm if side == "left" else right_arm

                    if arm_data:
                        # 映射关节名并更新位置
                        positions = {}
                        for key, value in arm_data.items():
                            joint_name = TRANSFORMER_TO_JOINT.get(key)
                            if joint_name is not None:
                                positions[joint_name] = value

                        self.update_joint_positions(positions)

                self.scene.step()
                self.viewer.render()

                # 显示摄像头画面
                display = frame.copy()
                cv2.putText(display, f"Joint Spheres: {side} arm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 绘制颜色图例
                self.draw_color_legend(display)

                cv2.imshow("Camera", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            print("清理资源...")
            detector.stop()
            cap.release()
            cv2.destroyAllWindows()

    def run_interactive(self):
        """运行交互式可视化"""
        print("\n=== Sapien 交互模式 ===")
        print("鼠标操作: 左键旋转, 右键平移, 滚轮缩放")
        print("关闭窗口退出")
        print("=" * 30)

        # 放置一些示例球体用于测试
        test_positions = {
            "SHOULDER": [0, 0, 0],
            "ELBOW": [0.1, 0, -0.15],
            "WRIST": [0.2, 0, -0.3],
            "THUMB_CMC": [0.22, 0.02, -0.32],
            "THUMB_MCP": [0.24, 0.04, -0.34],
            "THUMB_IP": [0.26, 0.05, -0.36],
            "THUMB_TIP": [0.28, 0.06, -0.38],
            "INDEX_FINGER_MCP": [0.25, 0, -0.35],
            "INDEX_FINGER_PIP": [0.28, 0, -0.38],
            "INDEX_FINGER_DIP": [0.30, 0, -0.40],
            "INDEX_FINGER_TIP": [0.32, 0, -0.42],
        }
        self.update_joint_positions(test_positions)

        while not self.viewer.closed:
            self.scene.step()
            self.viewer.render()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sapien 关节可视化器")
    parser.add_argument("--side", type=str, default="left",
                        choices=["left", "right"],
                        help="手臂侧（left/right）")
    parser.add_argument("--mode", type=str, default="detect",
                        choices=["detect", "interactive"],
                        help="运行模式: detect=实时检测, interactive=交互")
    parser.add_argument("--radius", type=float, default=0.015,
                        help="关节圆球半径（默认0.015）")

    args = parser.parse_args()

    vis = JointVisualizer(sphere_radius=args.radius)
    vis.setup_scene()
    vis.create_joint_spheres()
    vis.create_viewer()

    try:
        if args.mode == "detect":
            vis.run_with_detection(side=args.side)
        else:
            vis.run_interactive()
    finally:
        vis.viewer.close()


if __name__ == "__main__":
    main()
