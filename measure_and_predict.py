import cv2
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler from disk
bmodel = joblib.load("brightness_linear_regression_model.joblib")
dmodel = joblib.load("denoising_linear_regression_model.joblib")


def measure_frame_properties(image):
    # Ensure the image is in the correct format (8-bit grayscale)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Measure Brightness
    brightness = np.mean(gray)
    # Measure Noise (using the standard deviation of the Laplacian)
    noise = cv2.Laplacian(gray, cv2.CV_64F).var()
    return brightness, noise


def predictor(frame):
    # Create a DataFrame from the input frame
    frame_df = pd.DataFrame([frame], columns=["Brightness", "Denoising Value"])

    # Ensure the input to predict method has valid feature names
    brightness_feature = pd.DataFrame(frame_df["Brightness"], columns=["Brightness"])
    denoising_feature = pd.DataFrame(
        frame_df["Denoising Value"], columns=["Denoising Value"]
    )

    # Make predictions using the respective models
    brightness_prediction = bmodel.predict(brightness_feature)[0]
    denoising_prediction = dmodel.predict(denoising_feature)[0]

    # Return the predictions as a list
    return [brightness_prediction, denoising_prediction]


# image_path = r"E:\LLODProj1\bus.jpg"
# image = cv2.imread(image_path)
# frame = measure_frame_properties(image)
# # print(frame)
# print(predictor(frame))
