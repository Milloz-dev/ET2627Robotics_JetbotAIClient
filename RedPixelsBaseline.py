import os
import cv2
import numpy as np

"""Calculate Red pixle Baselines for each distance"""

DATASET_PATH = "RobotCode/jetbot_photos/redbox_distance"
LABELS = ["20cm", "40cm", "60cm", "over60cm"]

def count_red_pixels(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red wraps around HSV, so we use two ranges
    mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))

    mask = mask1 + mask2

    red_pixels = np.sum(mask > 0)

    return red_pixels

baselines = {}

for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)

    values = []

    for name in ["front.jpg", "side.jpg"]:
        path = os.path.join(folder, name)

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        img = cv2.imread(path)
        red_count = count_red_pixels(img)

        print(label, name, "red pixels:", red_count)
        values.append(red_count)

    if values:
        baselines[label] = np.mean(values)

print("\nBaselines:")
for label, value in baselines.items():
    print(label, int(value))

#Baselines
# 20cm front.jpg red pixels: 13999
# 20cm side.jpg red pixels: 10480
# 40cm front.jpg red pixels: 4966
# 40cm side.jpg red pixels: 4224
# 60cm front.jpg red pixels: 2233
# 60cm side.jpg red pixels: 1422
# over60cm front.jpg red pixels: 922
# over60cm side.jpg red pixels: 486

# Baselines:
# 20cm 12239
# 40cm 4595
# 60cm 1827
# over60cm 704