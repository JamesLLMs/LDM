import cv2
import os

def mirror_video(input_path, output_path):
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return

    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec

    # Create video writer object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_path} -> {output_path}")

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Horizontal flip (1 for horizontal, 0 for vertical, -1 for both)
        mirrored_frame = cv2.flip(frame, 1)

        # Write frame
        out.write(mirrored_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    # Release resources
    cap.release()
    out.release()
    print("Video mirroring completed!")

if __name__ == "__main__":
    input_video = "example/vector_retargeting/myrecord.mp4"
    output_video = "example/vector_retargeting/myrecord_mirrored.mp4"
    mirror_video(input_video, output_video)
