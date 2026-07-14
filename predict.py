import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model


# Paths
CNN_PATH = "../Models/bike_cnn_model.h5"
SVM_PATH = "../Models/bike_svm_model.pkl"
ENCODER_PATH = "../Models/label_encoder.pkl"

IMAGE_PATH = "../Dataset/test.jpg"


# Load CNN model
cnn_model = load_model(CNN_PATH)


# Remove final classification layer and use CNN as feature extractor
feature_model = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-2].output
)


# Load SVM
svm = pickle.load(open(SVM_PATH, "rb"))

# Load label encoder
encoder = pickle.load(open(ENCODER_PATH, "rb"))


# Load image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found!")
    exit()


# Preprocess image
img = cv2.resize(img, (128,128))

img = img / 255.0

img = np.expand_dims(img, axis=0)


# Extract CNN features
features = feature_model.predict(img)


# SVM Prediction
prediction = svm.predict(features)[0]


# Confidence
probabilities = svm.predict_proba(features)

confidence = max(probabilities[0]) * 100


# Convert label number back to bike name
bike_name = encoder.inverse_transform([prediction])


print("----------------------------")
print("Predicted Bike Model:", bike_name[0])
print("Confidence:", round(confidence,2), "%")
print("----------------------------")
