import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import pickle
from PIL import Image, ImageTk

from tensorflow.keras.models import load_model, Model


# Load CNN Model
cnn_model = load_model("../Models/bike_cnn_model.h5")

# Create Feature Extractor
feature_model = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-2].outputS
)

# Load SVM and Label Encoder
svm = pickle.load(open("../Models/bike_svm_model.pkl", "rb"))
encoder = pickle.load(open("../Models/label_encoder.pkl", "rb"))


def select_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if file_path:

        # Show selected image
        img_display = Image.open(file_path)
        img_display = img_display.resize((250, 250))

        photo = ImageTk.PhotoImage(img_display)

        image_label.config(image=photo)
        image_label.image = photo

        result_label.config(text="Processing...")

        # Read image
        img = cv2.imread(file_path)

        # Resize image
        img = cv2.resize(img, (128, 128))

        # Normalize
        img = img / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Extract CNN Features
        features = feature_model.predict(img, verbose=0)

        # Get probabilities
        probabilities = svm.predict_proba(features)[0]

        # Top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]

        result_text = "Prediction Results\n\n"

        for idx in top3_indices:

            bike = encoder.inverse_transform([idx])[0]

            score = probabilities[idx] * 100

            result_text += f"{bike} : {score:.2f}%\n"

        result_label.config(text=result_text)


# Create GUI Window
window = tk.Tk()

window.title("Bike Model Detection System")

window.geometry("800x700")


# Title
title = tk.Label(
    window,
    text="Deep Learning Based Bike Model Detection System",
    font=("Arial", 18, "bold")
)

title.pack(pady=20)


# Upload Button
upload_btn = tk.Button(
    window,
    text="Select Bike Image",
    command=select_image,
    font=("Arial", 12)
)

upload_btn.pack(pady=20)


# Image Preview Area
image_label = tk.Label(
    window,
    bd=2,
    relief="solid"
)

image_label.pack(pady=10)


# Result Area
result_label = tk.Label(
    window,
    text="Select a bike image",
    font=("Arial", 14),
    justify="left"
)

result_label.pack(pady=30)


window.mainloop()
