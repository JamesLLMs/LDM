# LDM (Lightweight Dexterous Motion)

![Demo](Pick.gif)

This repository contains experiments and utilities for **Arms (including hands) keypoint detection** and **motion retargeting** from **monocular video or a live camera**. It estimates upper-body and hand keypoints, applies multi-stage coordinate transforms, and retargets the motion to a robot **URDF** model, with visualization and debugging tools.

## Features

- **Video / webcam input**: Process frames from a local `mp4` or a live camera.
- **Body + hand keypoints**: Fuse pose and hand keypoints into joint / vector signals used for retargeting.
- **Multi-stage coordinate transforms**: Transform from detector coordinates to a first-person convention and then to an URDF-standard convention to help validate coordinate definitions.
- **Retargeting optimization**: Map human-side constraints (vectors / joints) into the robot joint space and output robot joint targets.
- **Visualization & debugging**: 2D video preview and 3D (VPython) coordinate / skeleton visualization scripts.

## Hand Retargeting Results (GIF)

| Part 1 | Part 2 |
| --- | --- |
| ![](example/vector_retargeting/result/compressed_video_part_1.gif) | ![](example/vector_retargeting/result/compressed_video_part_2.gif) |

| Part 3 | Part 4 |
| --- | --- |
| ![](example/vector_retargeting/result/compressed_video_part_3.gif) | ![](example/vector_retargeting/result/compressed_video_part_4.gif) |

## Repository Layout (High Level)

- `example/vector_retargeting/`: Example scripts and assets (e.g., wholebody video retargeting pipeline).
- `src/whobody_dect/`: Detection and visualization utilities (e.g., multi-stage coordinate visualization).
- `src/dex_retargeting/`: Retargeting and optimization implementation.

> Note: The primary entry points are under `example/`. Please check script arguments and local path configuration when running.

## Environment & Dependencies

Dependencies vary by script. Common requirements include (but are not limited to):

- Python 3
- `opencv-python`
- `numpy`
- `tyro` (if an entry point uses `tyro.cli`)
- `vpython` (for 3D visualization)
- `sapien` (for URDF loading / simulation, if you use robot visualization or simulation)

Install missing dependencies based on runtime errors (e.g., via `pip install ...`).

## Quick Start

### 1) Run the wholebody retargeting example

- Use webcam (when `video_path` is empty):

```bash
python3 example/vector_retargeting/retarget_from_wholebody_video.py
```

- Use a local video file (pass `--video_path` as required by the script):

```bash
python3 example/vector_retargeting/retarget_from_wholebody_video.py --video_path path/to/your/video.mp4
```

> If you run inside Docker/containers and want to access the webcam, make sure `/dev/video*` is mapped into the container and you have sufficient permissions.

### 2) Visualize / debug coordinate transforms

To validate the transform chain (detector coordinates -> first-person -> URDF standard):

```bash
python3 src/whobody_dect/simple_visualize.py
```

### 3) Mirror (flip) a video horizontally

If you need a mirrored version of a video (e.g., to match selfie orientation), you can generate a flipped output using OpenCV:

```bash
python3 src/whobody_dect/mirror_video.py
```

Default input: `example/vector_retargeting/myrecord.mp4`

Default output: `example/vector_retargeting/myrecord_mirrored.mp4`

## Git Push Notes (Based on Your Current Remote Setup)

Your `git remote -v` indicates:

- `mer_wholebody` points to your repository: `https://github.com/JamesLLMs/LDM.git`
- `origin` points to the upstream repository: `https://github.com/dexsuite/dex-retargeting`

To push your local `main` to your own repository:

```bash
git push -u mer_wholebody main
```

If you prefer using `git push` without specifying a remote each time, you may rename `mer_wholebody` to `origin` (be careful if you still want to keep the upstream `origin`):

```bash
git remote remove origin
git remote rename mer_wholebody origin
git push -u origin main
```

## Credits / References

This project is **inspired by** and **partially organized with reference to** the following open-source projects and tools (many thanks):

- **dex-retargeting**: The retargeting approach and parts of the engineering structure in this repository reference this project.
  - Upstream: <https://github.com/dexsuite/dex-retargeting>
- **SAPIEN**: Robot URDF loading and simulation utilities.
  - <https://sapien.ucsd.edu/>
- **OpenCV**: Video I/O and visualization.
  - <https://opencv.org/>
- **VPython**: 3D coordinate and skeleton visualization.
  - <https://vpython.org/>

> If you use additional detectors or fusion modules (e.g., MediaPipe or custom models), consider adding their references here as well.
