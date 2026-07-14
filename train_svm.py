import os
import cv2
import numpy as np
import pickle

from tensorflow.keras.models import load_model, Model
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Paths
DATASET_PATH = "../Dataset"
MODEL_PATH = "../Models/bike_cnn_model.h5"
SVM_PATH = "../Models/bike_svm_model.pkl"


# Load CNN model
cnn_model = load_model(MODEL_PATH)

feature_model = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-2].output
)

print("CNN Feature Extractor Loaded")


images = []
labels = []

IMG_SIZE = 128


# Load dataset
for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(DATASET_PATH, category)

    if os.path.isdir(category_path):

        for img_name in os.listdir(category_path):

            img_path = os.path.join(category_path, img_name)

            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                images.append(img)
                labels.append(category)


images = np.array(images)
labels = np.array(labels)


# Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)


print("Extracting CNN Features...")


features = feature_model.predict(images)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels_encoded,
    test_size=0.2,
    random_state=42
)


# Train SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)


# Accuracy
prediction = svm.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("SVM Accuracy:", accuracy)


# Save SVM model
pickle.dump(
    svm,
    open(SVM_PATH, "wb")
)

pickle.dump(
    encoder,
    open("../Models/label_encoder.pkl", "wb")
)


print("SVM Model Saved Successfully")
