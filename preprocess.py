from moviepy.editor import *
import cv2


# Function to enhance low-light video
def enhance_video(input_video_path):
    clip = (
        VideoFileClip(input_video_path)
        .fx(vfx.gamma_corr, gamma=0.5)
        .fx(vfx.colorx, factor=1.2)
    )
    return clip


def denoise_video(input_video_path, output_video_path):
    # Create video capture object
    cap = cv2.VideoCapture(input_video_path)

    # Check if video opening was successful
    if not cap.isOpened():
        print(f"Error opening video: {input_video_path}")
        return None

    # Get frame rate from original video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer with appropriate codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Adjust codec if necessary
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Check if video writer creation was successful
    if not out.isOpened():
        print(f"Error opening output video: {output_video_path}")
        cap.release()
        return None

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Apply non-local means denoising (adjust parameters as needed)
        denoised_frame = cv2.fastNlMeansDenoising(frame, None, 11, 11, 10)

        # Write denoised frame to output video
        out.write(denoised_frame)

    # Release resources
    cap.release()
    out.release()

    # Return path of denoised video file
    return output_video_path
