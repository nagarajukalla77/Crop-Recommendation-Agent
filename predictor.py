import joblib
import numpy as np

# Load trained model
model = joblib.load("models/best_crop_model.pkl")

# Load label encoder
label_encoder = joblib.load("models/label_encoder.pkl")


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict crop using the trained Random Forest model.
    """

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(features)

    crop = label_encoder.inverse_transform(prediction)[0]

    return crop