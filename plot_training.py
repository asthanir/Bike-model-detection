import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

epochs = [1,2,3,4,5,6,7,8,9,10]

accuracy = [
    0.3231,
    0.3231,
    0.3231,
    0.5077,
    0.6205,
    0.6051,
    0.8821,
    0.9231,
    0.9179,
    0.9692
]

val_accuracy = [
    0.1633,
    0.1633,
    0.1633,
    0.3469,
    0.2449,
    0.3061,
    0.4082,
    0.4286,
    0.3673,
    0.4286
]

loss = [
    1.5801,
    1.5801,
    1.5801,
    1.4223,
    1.0596,
    1.0546,
    0.5787,
    0.3294,
    0.2174,
    0.1576
]

val_loss = [
    1.6114,
    1.6114,
    1.6114,
    1.5116,
    2.4237,
    1.3780,
    1.5347,
    1.6068,
    1.7549,
    1.7072
]

# Accuracy graph
plt.figure(figsize=(8,5))
plt.plot(epochs, accuracy, label="Training Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("../Outputs/cnn_accuracy.png")
plt.close()

# Loss graph
plt.figure(figsize=(8,5))
plt.plot(epochs, loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("../Outputs/cnn_loss.png")
plt.close()

print("Graphs saved successfully")
