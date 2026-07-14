import os
import cv2
import numpy as np

dataset_path = "../dataset"

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
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.0

                    images.append(img)
                    labels.append(folder)

print("Total Images Loaded:", len(images))
print("Total Labels:", len(labels))

images = np.array(images)

print("Image Shape:", images.shape)
