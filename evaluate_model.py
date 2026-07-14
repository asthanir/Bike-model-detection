import matplotlib
matplotlib.use("Agg")
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns


# Paths
DATASET_PATH = "../Dataset"
CNN_PATH = "../Models/bike_cnn_model.h5"
SVM_PATH = "../Models/bike_svm_model.pkl"
ENCODER_PATH = "../Models/label_encoder.pkl"


# Load models
cnn_model = load_model(CNN_PATH)

feature_model = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-2].output
)

svm = pickle.load(open(SVM_PATH, "rb"))
encoder = pickle.load(open(ENCODER_PATH, "rb"))


images = []
labels = []

IMG_SIZE = 128


# Load images
for category in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, category)

    if os.path.isdir(folder):

        for img_name in os.listdir(folder):

            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path)

            if img is not None:

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                images.append(img)
                labels.append(category)


images = np.array(images)
labels = np.array(labels)


# Encode labels
labels_encoded = encoder.transform(labels)


# Extract features
features = feature_model.predict(images)


# Prediction
prediction = svm.predict(features)


# Accuracy
accuracy = np.mean(prediction == labels_encoded)

print("Accuracy:", accuracy)


# Report
print(
    classification_report(
        labels_encoded,
        prediction,
        target_names=encoder.classes_
    )
)


# Confusion matrix
cm = confusion_matrix(
    labels_encoded,
    prediction
)


plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Bike Model Confusion Matrix")

plt.savefig("../Outputs/confusion_matrix.png")

# plt.show()
