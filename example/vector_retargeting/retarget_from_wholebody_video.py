"""
Whole-body retargeting main program

Detects human arm and hand poses from video and retargets to robot arm + robot hand.

Supported robots:
    - ur5e_shadow: UR5e robot arm + Shadow Hand
    - xarm7_ability: xArm7 robot arm + Ability Hand

Usage:
    python retarget_from_wholebody_video.py --video_path video.mp4 --hand_type Right
    python retarget_from_wholebody_video.py --config_path configs/xarm7_ability_right.yml --hand_type Right
    python retarget_from_wholebody_video.py --show_human_joints True  # Show human joints
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.retargeting_config import RetargetingConfig
from wholebody_detector import WholeBodyDetector
from human_joint_visualizer import HumanJointVisualizer, load_coordinate_transform_from_config

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main(
    config_path: str = "configs/ur5e_shadow_rightv1.yml",
    hand_type: str = "Right",
    video_path: str = "myrecord_mirrored.mp4",
    is_selfie: bool = False,
    show_origin_frame: bool = True,
    show_human_joints: bool = True,
    human_joints_offset: tuple = (0.0, 0.0, 1.0),
):
    """
    Detect human arm pose from video and retarget to robot arm + robot hand.

    Args:
        config_path: Retargeting configuration file path (relative to current directory)
        hand_type: Hand type, "Left" or "Right"
        video_path: Video file path
        is_selfie: Whether to use selfie mode
        show_origin_frame: Whether to show coordinate system origin
        show_human_joints: Whether to show human joints (spheres)
        human_joints_offset: Human joint display position offset (x, y, z)
    """
    # ========================================
    # 1. Load retargeting configuration
    # ========================================
    config_file = Path(__file__).parent / config_path
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return

    # Set URDF directory to assets/robots
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    logger.info(f"Loading config from: {config_file}")
    retargeting = RetargetingConfig.load_from_file(config_file)
    retargeting = retargeting.build()

    # ========================================
    # 2. Initialize detector
    # ========================================
    logger.info(f"Initializing WholeBodyDetector for {hand_type} hand...")
    detector = WholeBodyDetector(
        hand_type=hand_type,
        is_selfie=is_selfie,
    )

    # ========================================
    # 3. Setup SAPIEN scene
    # ========================================
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    scene = sapien.Scene()

    # Ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.5, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )

    # Camera - adjust position to fit larger robot arm model
    cam = scene.add_camera(
        name="video_cam", width=800, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.0, 0.0, 1.2], [0.0, 0.0, 0.0, 0.0]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = show_origin_frame
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # ========================================
    # 4. Create human joint visualizer
    # ========================================
    human_visualizer = None
    if show_human_joints:
        logger.info("Creating human joint visualizer...")
        # Load quaternion from config file (to set sphere pose, aligning axes with robot arm)
        rotation_quaternion = None
        transform_result = load_coordinate_transform_from_config("human_config.json")
        if transform_result is not None:
            _, rotation_quaternion = transform_result
            if rotation_quaternion is not None:
                logger.info(f"Loaded rotation quaternion: {rotation_quaternion}")

        human_visualizer = HumanJointVisualizer(
            scene=scene,
            sphere_radius=0.015,
            position_offset=np.array(human_joints_offset),
            rotation_quaternion=rotation_quaternion,
            visible=True,
        )
        human_visualizer.create_joint_spheres()
        logger.info(f"Human joint visualizer created with {human_visualizer.num_joints} joints")

    # ========================================
    # 5. Load robot model
    # ========================================
    config = RetargetingConfig.load_from_file(config_file)
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    loader.load_multiple_collisions_from_file = True

    urdf_path = str(filepath)
    if "glb" not in filepath.stem:
        urdf_path = urdf_path.replace(".urdf", "_glb.urdf")

    logger.info(f"Loading robot from: {urdf_path}")
    robot = loader.load(urdf_path)
    robot.set_pose(sapien.Pose([0, 0, 1]))  # Robot arm base at origin

    # Joint mapping
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    logger.info(f"SAPIEN joints: {sapien_joint_names}")
    logger.info(f"Retargeting joints: {retargeting_joint_names}")

    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # ========================================
    # 6. Video processing loop
    # ========================================
    # Use default camera when video_path is empty, otherwise use video file path
    if video_path:
        video_file = str(Path(__file__).parent / video_path)
    else:
        video_file = 2

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_file}")
        return

    logger.info("=" * 50)
    logger.info("Starting wholebody retargeting...")
    logger.info(f"Hand type: {hand_type}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Video: {video_path}")
    logger.info(f"Show human joints: {show_human_joints}")
    logger.info("Press 'q' to quit, 'h' to toggle human joints visibility")
    logger.info("=" * 50)

    frame_count = 0
    while cap.isOpened():
        success, bgr = cap.read()
        if not success:
            time.sleep(1 / 30.0)
            cap.release()
            logger.debug("Restarting video from the beginning")
            cap = cv2.VideoCapture(str(video_file))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            logger.info("End of video file reached.")
            continue

        frame_count += 1
        logger.debug(frame_count)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        num_detected, joint_pos, _, _ = detector.detect(rgb)

        # Display video
        cv2.imshow("wholebody_retargeting", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            continue
        elif key == ord("h") and human_visualizer is not None:
            # Toggle human joint visibility
            human_visualizer.set_visible(not human_visualizer.visible)
            #logger.info(f"Human joints visibility: {human_visualizer.visible}")
        # ========================================
        # 7. Update human joint visualization
        # ========================================
        if joint_pos is None or len(joint_pos) != 24:
            #logger.warning(f"{hand_type} arm is not detected in this frame.")
            print(f" joint is not detected in this frame.--------------------------------")
            continue
        else :
            #logger.debug(f"{joint_pos} arm is detected in this frame.")
            if joint_pos[2][0] == [0]:
                print(f"joint_pos[2][0] is [0]--------------------------------")
                continue
            else:
                print(joint_pos[2][0])


        if human_visualizer is not None:
            human_visualizer.update_from_array(joint_pos)
        # ========================================
        # 8. Compute retargeting
        # ========================================
        # Get vector indices
        indices = retargeting.optimizer.target_link_human_indices
        origin_indices = indices[0, :]
        task_indices = indices[1, :]

        # Calculate reference vectors
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

        # Execute retargeting
        qpos = retargeting.retarget(ref_value, fixed_qpos=np.array([ 0,0]))
        robot.set_qpos(qpos[retargeting_to_sapien])

        for _ in range(2):
            viewer.render()

    # ========================================
    # 9. Cleanup
    # ========================================
    cap.release()
    cv2.destroyAllWindows()
    detector.stop()
    logger.info("Done")


if __name__ == "__main__":
    tyro.cli(main(video_path="myrecord_mirrored.mp4"))
