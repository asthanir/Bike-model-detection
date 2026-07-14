import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_path = "../Dataset"

images = []
labels = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):

            if file.lower().endswith(('.jpg', '.jpeg', '.png')):

                img_path = os.path.join(folder_path, file)

                img = cv2.imread(img_path)

                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    img = img / 255.0

                    images.append(img)
                    labels.append(folder)

X = np.array(images)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

model.save("../Models/bike_cnn_model.h5")
