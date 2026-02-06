#!/usr/bin/env python3
"""
Simplified multi-stage coordinate transformation visualization

Specifically test if coordinate conversion is correct, showing:
1. Mediapipe original coordinates
2. First stage conversion (first-person perspective)
3. Second stage conversion (URDF standard)
"""

import cv2
from vpython import *
import numpy as np
from fusion_module import PoseHandFusion
from arm_coordinate_transformer import ArmCoordinateTransformer


def create_simple_scenes():
    """Create three simplified visualization scenes"""
    scene1 = canvas(title="1. Mediapipe Raw", width=500, height=400)
    scene1.background = vector(0.2, 0.2, 0.3)  # Dark gray-blue background
    scene1.camera.pos = vector(0, 0, 2)
    scene1.center = vector(0, 0, 0)

    scene2 = canvas(title="2. First-Person", width=500, height=400)
    scene2.background = vector(0.2, 0.2, 0.3)
    scene2.camera.pos = vector(0, 0, 2)
    scene2.center = vector(0, 0, 0)

    scene3 = canvas(title="3. URDF Standard", width=500, height=400)
    scene3.background = vector(0.2, 0.2, 0.3)
    scene3.camera.pos = vector(0, 0, 2)
    scene3.center = vector(0, 0, 0)

    return scene1, scene2, scene3


def draw_axes_simple(scene, stage_type):
    """Draw simple coordinate axes"""
    scene.select()
    scale = 1.2

    if stage_type == "world":
        # World coordinates: X right, Y up, Z forward
        arrow(pos=vector(0, 0, 0), axis=vector(scale, 0, 0), color=color.red)
        arrow(pos=vector(0, 0, 0), axis=vector(0, scale, 0), color=color.green)
        arrow(pos=vector(0, 0, 0), axis=vector(0, 0, scale), color=color.blue)
    elif stage_type == "fp":
        # First-person: X right, Y down, Z forward
        arrow(pos=vector(0, 0, 0), axis=vector(scale, 0, 0), color=color.red)
        arrow(pos=vector(0, 0, 0), axis=vector(0, scale, 0), color=color.green)
        arrow(pos=vector(0, 0, 0), axis=vector(0, 0, -scale), color=color.blue)
    else:  # urdf
        # URDF: X forward, Y left, Z up
        arrow(pos=vector(0, 0, 0), axis=vector(scale, 0, 0), color=color.red)
        arrow(pos=vector(0, 0, 0), axis=vector(0, scale, 0), color=color.green)
        arrow(pos=vector(0, 0, 0), axis=vector(0, 0, scale), color=color.blue)


def draw_points_and_lines(scene, joints_dict, stage_type, points_pool, lines_pool):
    """Draw points and connecting lines"""
    scene.select()

    if not joints_dict:
        return

    # Get main joints
    key_joints = ["SHOULDER", "ELBOW", "WRIST"]
    hand_joints = [k for k in joints_dict.keys() if "HAND_" in k]

    # Draw points
    active = set()
    for joint_name in key_joints + hand_joints:
        if joint_name not in joints_dict:
            continue
        active.add(joint_name)
        x, y, z = joints_dict[joint_name]

        # Coordinate transformation
        if stage_type == "world":
            # World coordinates directly use
            vec = vector(x, y, z)
        elif stage_type == "fp":
            # First-person perspective: X right, Y down, Z forward -> VPython: X right, Y up, Z back
            vec = vector(x, -y, -z)
        else:  # urdf
            # URDF: X forward, Y left, Z up -> VPython: X right, Y up, Z up
            vec = vector(x, -y, z)

        # Set color and size
        if joint_name == "SHOULDER":
            color_obj = color.red
            radius = 0.05
        elif joint_name == "ELBOW":
            color_obj = color.green
            radius = 0.04
        elif joint_name == "WRIST" or joint_name == "HAND_WRIST":
            color_obj = color.blue
            radius = 0.02
        elif "THUMB" in joint_name:
            color_obj = color.orange
            radius = 0.008
        elif "INDEX" in joint_name:
            color_obj = color.yellow
            radius = 0.008
        elif "MIDDLE" in joint_name:
            color_obj = color.cyan
            radius = 0.008
        elif "RING" in joint_name:
            color_obj = color.magenta
            radius = 0.008
        elif "PINKY" in joint_name:
            color_obj = vector(0.5, 1, 0.5)  # Light green
            radius = 0.008
        else:
            color_obj = color.white
            radius = 0.01

        if joint_name not in points_pool:
            points_pool[joint_name] = sphere(pos=vec, radius=radius, color=color_obj, canvas=scene)
        else:
            points_pool[joint_name].pos = vec
            points_pool[joint_name].visible = True

    # Hide disappeared points
    for name, obj in points_pool.items():
        if name not in active:
            obj.visible = False

    # Complete bone connection relationships (including hand)
    connections = [
        # Main arm chain
        ["SHOULDER", "ELBOW"],
        ["ELBOW", "WRIST"],
        # Thumb
        ["WRIST", "HAND_THUMB_CMC"],
        ["HAND_THUMB_CMC", "HAND_THUMB_MCP"],
        ["HAND_THUMB_MCP", "HAND_THUMB_IP"],
        ["HAND_THUMB_IP", "HAND_THUMB_TIP"],
        # Index finger
        ["WRIST", "HAND_INDEX_FINGER_MCP"],
        ["HAND_INDEX_FINGER_MCP", "HAND_INDEX_FINGER_PIP"],
        ["HAND_INDEX_FINGER_PIP", "HAND_INDEX_FINGER_DIP"],
        ["HAND_INDEX_FINGER_DIP", "HAND_INDEX_FINGER_TIP"],
        # MCP chain + Middle finger
        ["HAND_INDEX_FINGER_MCP", "HAND_MIDDLE_FINGER_MCP"],
        ["HAND_MIDDLE_FINGER_MCP", "HAND_MIDDLE_FINGER_PIP"],
        ["HAND_MIDDLE_FINGER_PIP", "HAND_MIDDLE_FINGER_DIP"],
        ["HAND_MIDDLE_FINGER_DIP", "HAND_MIDDLE_FINGER_TIP"],
        # MCP chain + Ring finger
        ["HAND_MIDDLE_FINGER_MCP", "HAND_RING_FINGER_MCP"],
        ["HAND_RING_FINGER_MCP", "HAND_RING_FINGER_PIP"],
        ["HAND_RING_FINGER_PIP", "HAND_RING_FINGER_DIP"],
        ["HAND_RING_FINGER_DIP", "HAND_RING_FINGER_TIP"],
        # MCP chain + Pinky finger
        ["HAND_RING_FINGER_MCP", "HAND_PINKY_MCP"],
        ["HAND_PINKY_MCP", "HAND_PINKY_PIP"],
        ["HAND_PINKY_PIP", "HAND_PINKY_DIP"],
        ["HAND_PINKY_DIP", "HAND_PINKY_TIP"],
        # Wrist to pinky MCP (metacarpal connection, optional)
        ["WRIST", "HAND_PINKY_MCP"],
    ]

    # Draw connecting lines
    line_idx = 0
    for start, end in connections:
        # Handle WRIST/HAND_WRIST mapping
        start_key = start if start in joints_dict else ("HAND_WRIST" if start == "WRIST" else start)
        end_key = end if end in joints_dict else ("HAND_WRIST" if end == "WRIST" else end)

        if start_key in joints_dict and end_key in joints_dict:
            if start_key in points_pool and end_key in points_pool:
                p1 = points_pool[start_key].pos
                p2 = points_pool[end_key].pos
                dist = mag(p2 - p1)

                if dist > 0.001:
                    # Set color based on link type
                    if "THUMB" in start or "THUMB" in end:
                        line_color = color.orange
                    elif "INDEX" in start or "INDEX" in end:
                        line_color = color.yellow
                    elif "MIDDLE" in start or "MIDDLE" in end:
                        line_color = color.cyan
                    elif "RING" in start or "RING" in end:
                        line_color = color.magenta
                    elif "PINKY" in start or "PINKY" in end:
                        line_color = vector(0.5, 1, 0.5)
                    else:
                        line_color = color.white  # Main arm chain uses white

                    if line_idx < len(lines_pool):
                        cyl = lines_pool[line_idx]
                        cyl.pos = p1
                        cyl.axis = p2 - p1
                        cyl.color = line_color
                        cyl.visible = True
                    else:
                        cyl = cylinder(pos=p1, axis=p2 - p1, radius=0.005,
                                      color=line_color, canvas=scene)
                        lines_pool.append(cyl)
                    line_idx += 1

    # Hide extra lines
    for i in range(line_idx, len(lines_pool)):
        lines_pool[i].visible = False


def main():
    # Create detector and transformers
    detector = PoseHandFusion("skeleton_config.json", is_selfie=False)
    transformer_fp = ArmCoordinateTransformer(use_urdf_standard=False)
    transformer_urdf = ArmCoordinateTransformer(use_urdf_standard=True)

    # Create scenes
    scene1, scene2, scene3 = create_simple_scenes()

    # Draw coordinate axes
    draw_axes_simple(scene1, "world")
    draw_axes_simple(scene2, "fp")
    draw_axes_simple(scene3, "urdf")

    # Create object pools
    stage1_objs = {"points": {}, "lines": []}
    stage2_objs = {"points": {}, "lines": []}
    stage3_objs = {"points": {}, "lines": []}

    cap = cv2.VideoCapture(0)

    print("=" * 60)
    print("Simplified Multi-Stage Coordinate Transformation Visualization")
    print("=" * 60)
    print("Window 1: Mediapipe Original World Coordinates")
    print("Window 2: First-Person Perspective Local Coordinates")
    print("Window 3: URDF Standard Coordinates")
    print("\nPress 'q' to quit")
    print("=" * 60)

    visual_path = "src/whobody_dect/myrecord_mirrored.mp4"
    visual_cap = cv2.VideoCapture(visual_path)
    if not visual_cap.isOpened():
        print(f"Failed to open visual video file: {visual_path}")
        return

    try:
        while True:
            rate(30)
            success, frame = visual_cap.read()
            if not success:
                break

            # Detect joints
            joints_dict, _ = detector.process_frame(frame)

            if joints_dict:
                # Stage 1: Original coordinates
                stage1_joints = {}
                for key in joints_dict:
                    if any(x in key for x in ["SHOULDER", "ELBOW", "WRIST", "HAND_"]):
                        stage1_joints[key] = joints_dict[key]

                if stage1_joints:
                    draw_points_and_lines(scene1, stage1_joints, "world",
                                        stage1_objs["points"], stage1_objs["lines"])

                # Stage 2: First-person perspective
                left_fp, right_fp = transformer_fp.compute_local_coordinates(joints_dict)
                if left_fp:
                    draw_points_and_lines(scene2, left_fp, "fp",
                                        stage2_objs["points"], stage2_objs["lines"])

                # Stage 3: URDF standard
                left_urdf, right_urdf = transformer_urdf.compute_local_coordinates(joints_dict)
                if left_urdf:
                    draw_points_and_lines(scene3, left_urdf, "urdf",
                                        stage3_objs["points"], stage3_objs["lines"])

            # Display debug information
            display_frame = frame.copy()
            cv2.putText(display_frame, "3-Stage Coordinate Visualization", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display coordinate information
            if joints_dict and "LEFT_ELBOW" in joints_dict:
                elbow = joints_dict["LEFT_ELBOW"]
                cv2.putText(display_frame, f"Raw: ({elbow[0]:.3f}, {elbow[1]:.3f}, {elbow[2]:.3f})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            if 'left_fp' in locals() and left_fp and "ELBOW" in left_fp:
                elbow = left_fp["ELBOW"]
                cv2.putText(display_frame, f"FP: ({elbow[0]:.3f}, {elbow[1]:.3f}, {elbow[2]:.3f})",
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            if 'left_urdf' in locals() and left_urdf and "ELBOW" in left_urdf:
                elbow = left_urdf["ELBOW"]
                cv2.putText(display_frame, f"URDF: ({elbow[0]:.3f}, {elbow[1]:.3f}, {elbow[2]:.3f})",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.imshow('Debug', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up resources...")
        detector.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
