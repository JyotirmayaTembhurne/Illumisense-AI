import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data1 = pd.read_csv(r"E:\LLODProj1\Brightness Values.csv")
# Define feature and target names
brightness_feature_names = ["Brightness"]
brightness_target_name = "Predicted Brightness"
# Split the data into features (X) and target (y)
X1 = data1[brightness_feature_names]
y1 = data1[brightness_target_name]
# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)
# Create and train the linear regression model
brightness_model = LinearRegression()
brightness_model.fit(X1_train, y1_train)
# Make predictions on the test set
y1_pred = brightness_model.predict(X1_test)
# Evaluate the model
mse1 = mean_squared_error(y1_test, y1_pred)
print(f"Mean Squared Error: {mse1}")
# Save the trained model
model_path1 = r"E:\LLODProj1\brightness_linear_regression_model.joblib"
joblib.dump(brightness_model, model_path1)
print(f"Model saved to {model_path1}")


data2 = pd.read_csv(r"E:\LLODProj1\Denoising Values.csv")
# Define feature and target names
noise_feature_names = ["Denoising Value"]
noise_target_name = "Predicted Denoising Value"
# Split the data into features (X) and target (y)
X2 = data2[noise_feature_names]
y2 = data2[noise_target_name]
# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)
# Create and train the linear regression model
noise_model = LinearRegression()
noise_model.fit(X2_train, y2_train)
# Make predictions on the test set
y2_pred = noise_model.predict(X2_test)
# Evaluate the model
mse2 = mean_squared_error(y2_test, y2_pred)
print(f"Mean Squared Error: {mse2}")
# Save the trained model
model_path2 = r"E:\LLODProj1\denoising_linear_regression_model.joblib"
joblib.dump(noise_model, model_path2)
print(f"Model saved to {model_path2}")
