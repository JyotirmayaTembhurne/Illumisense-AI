import os
import cv2
import streamlit as st
from ultralytics import YOLO
import measure_and_predict


def adjust_frame_properties(frame, properties):
    brightness, noise = properties
    if brightness <= 90:
        # Brightness adjustment
        brightness_factor = brightness / 43
        brightened_frame = cv2.convertScaleAbs(
            frame, alpha=brightness_factor, beta=brightness / 3
        )
    elif 150 < brightness < 255:
        # Brightness adjustment
        brightness_factor = brightness / 255
        brightened_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
    elif brightness > 255:
        # Brightness adjustment
        brightness_factor = 0
        brightened_frame = cv2.convertScaleAbs(
            frame, alpha=brightness_factor, beta=-255
        )
    else:
        brightened_frame = frame

    # Noise reduction using Gaussian blur
    blur_kernel_size = max(
        1, int(noise // 10) * 2 + 1
    )  # Ensure kernel size is odd and at least 1
    final_frame = cv2.GaussianBlur(
        brightened_frame, (blur_kernel_size, blur_kernel_size), 0
    )

    return final_frame


def rt_yolov8(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open video capture device.")
        return
    # Create a placeholder for the video
    video_placeholder = st.empty()

    while True:
        ret, frame = cap.read()

        if ret:
            properties = measure_and_predict.measure_frame_properties(frame)
            adjustments = measure_and_predict.predictor(properties)
            processed_frame = adjust_frame_properties(frame, adjustments)
            results = model(processed_frame)
            if results:
                # Extract the first result (we assume there's only one result for each frame)
                img_with_boxes = results[0].plot()

                # Convert to JPG format for display
                _, buffer = cv2.imencode(".jpg", img_with_boxes)
                frame_display = buffer.tobytes()

                # Display the frame in Streamlit app
                video_placeholder.image(frame_display, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
