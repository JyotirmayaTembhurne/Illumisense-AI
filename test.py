import cv2
import numpy as np
import measure_and_predict


# # Define feature and target names
# feature_names = ["Brightness", "Gamma", "Denoising Value", "Contrast"]
# target_names = [
#     "Predicted Brightness",
#     "Predicted Gamma",
#     "Predicted Denoising Value",
#     "Predicted Contrast",
# ]


# def measure_video_properties(video_path):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Initialize lists to store the original and predicted properties for each frame
#     original_properties_list = []
#     predicted_properties_list = []

#     # Loop through the video frames
#     while cap.isOpened():
#         # Read a frame from the video
#         ret, frame = cap.read()

#         if ret:
#             # Measure properties for the current frame
#             brightness, gamma, noise, contrast = ml.measure_frame_properties(frame)

#             # Store the properties in a list for this frame
#             frame_properties = [brightness, gamma, noise, contrast]
#             predicted = ml.predictor(frame_properties)

#             # Append the properties of this frame to the respective lists
#             original_properties_list.append(frame_properties)
#             predicted_properties_list.append(predicted)
#         else:
#             # Break the loop if no more frames are available
#             break

#     # Release the video capture object
#     cap.release()

#     # Convert the lists of properties to numpy arrays
#     original_properties_array = np.array(original_properties_list)
#     predicted_properties_array = np.array(predicted_properties_list)

#     return original_properties_array, predicted_properties_array


# # Test the function
# original_properties, predicted_properties = measure_video_properties(
#     r"E:\LLODProj1\LLODVid1.mp4"
# )
# print("Original properties:", original_properties)
# print("Predicted properties:", predicted_properties)


def adjust_frame_properties(frame, properties):
    brightness, gamma, noise, contrast = properties

    # Brightness adjustment
    brightness_factor = brightness / 128  # Normalize brightness to range [0, 1]
    brightened_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    # Contrast adjustment
    f = 131 * (contrast + 127) / (127 * (131 - contrast))
    contrasted_frame = cv2.convertScaleAbs(
        brightened_frame, alpha=f, beta=127 * (1 - f)
    )

    # Gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    gamma_corrected_frame = cv2.LUT(contrasted_frame, table)

    # Noise reduction using Gaussian blur
    # Noise level is not directly used in Gaussian blur but let's assume more noise means more blur
    # We use a kernel size proportional to the noise value
    blur_kernel_size = max(
        1, int(noise // 10) * 2 + 1
    )  # Ensure kernel size is odd and at least 1
    final_frame = cv2.GaussianBlur(
        gamma_corrected_frame, (blur_kernel_size, blur_kernel_size), 0
    )

    return final_frame


# Example usage:
frame = cv2.imread(r"E:\LLODProj1\bus.jpg")
adjusted_frame = adjust_frame_properties(frame, [110, 0.7, 12.1, 30])
cv2.imshow("Adjusted Frame", adjusted_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
