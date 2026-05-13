import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATASET_PATH = "RobotCode/jetbot_photos/redbox_distance"

LABELS = ["20cm", "40cm", "60cm","over60cm"]

IMG_SIZE = 224


def load_images():
    X = []
    y = []

    for label in LABELS:
        folder = os.path.join(DATASET_PATH, label)

        if not os.path.exists(folder):
            print(f"Missing folder: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.startswith("."):
                continue

            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, filename)
            img = cv2.imread(path)

            if img is None:
                print("Could not read:", path)
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            features = img.flatten()

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


print("Loading images...")
X, y = load_images()

print("Images loaded:", len(X))

for label in LABELS:
    print(f"{label} images:", np.sum(y == label))

if len(set(y)) < len(LABELS):
    raise ValueError("Need images from all folders: " + str(LABELS))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = KNeighborsClassifier(n_neighbors=3)

print("Training...")
model.fit(X_train, y_train)

print("Testing...")
y_pred = model.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=LABELS))

print(classification_report(y_test, y_pred))

np.save("X_train.npy", X)
np.save("y_train.npy", y)

print("Saved X_train.npy and y_train.npy")