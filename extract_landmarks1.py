import os

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import csv
import json
import gc
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\ritik\Downloads\original_images"
OUTPUT_CSV = os.path.join(BASE_DIR, "landmarks_dataset11.csv")
MAPPING_SAVE_PATH = os.path.join(BASE_DIR, "class_mapping_mediapipe11.json")
HAND_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


def process_dataset():
    if not os.path.exists(HAND_TASK_PATH):
        print(f"❌ ERROR: Model file missing!");
        return
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset folder not found!");
        return

    try:
        base_options = python.BaseOptions(model_asset_path=HAND_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,  # ISL ke liye 2 haath zaroori hain
            min_hand_detection_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"❌ MediaPipe Init Error: {e}");
        return

    classes = sorted(os.listdir(DATASET_PATH))
    with open(MAPPING_SAVE_PATH, 'w') as f:
        json.dump({i: c for i, c in enumerate(classes)}, f)

    valid_samples = 0
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header for 126 features (21*3*2)
        header = ['label']
        for i in range(21): header.extend([f'l_x{i}', f'l_y{i}', f'l_z{i}'])
        for i in range(21): header.extend([f'r_x{i}', f'r_y{i}', f'r_z{i}'])
        writer.writerow(header)

        for label_idx, class_name in enumerate(classes):
            class_dir = os.path.join(DATASET_PATH, class_name)
            if not os.path.isdir(class_dir): continue

            print(f"Processing: {class_name}")
            images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                image_cv = cv2.imread(img_path)
                if image_cv is None: continue

                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = detector.detect(mp_image)

                if results.hand_landmarks:
                    # Logic: X-Coordinate ke basis pe hands ko sort karo
                    # Isse mirror effect ka masla khatam ho jayega
                    hands_list = []
                    for idx, hand_lms in enumerate(results.hand_landmarks):
                        # Wrist ka X coordinate save karo
                        hands_list.append((hand_lms[0].x, hand_lms))

                    # Sort hands by X (Jo screen ke left mein hai wo pehle)
                    hands_list.sort(key=lambda x: x[0])

                    left_hand_data = [0.0] * 63
                    right_hand_data = [0.0] * 63

                    for i, (wrist_x, hand_lms) in enumerate(hands_list):
                        if i >= 2: break  # Sirf 2 haath max

                        # Normalization
                        base = hand_lms[0]
                        max_dist = 0
                        for lm in hand_lms:
                            d = ((lm.x - base.x) ** 2 + (lm.y - base.y) ** 2 + (lm.z - base.z) ** 2) ** 0.5
                            if d > max_dist: max_dist = d
                        if max_dist == 0: max_dist = 1

                        temp_data = []
                        for lm in hand_lms:
                            temp_data.extend(
                                [(lm.x - base.x) / max_dist, (lm.y - base.y) / max_dist, (lm.z - base.z) / max_dist])

                        # Pehla sorted haath Left column mein, doosra Right column mein
                        if i == 0:
                            left_hand_data = temp_data
                        else:
                            right_hand_data = temp_data

                    writer.writerow([label_idx] + left_hand_data + right_hand_data)
                    valid_samples += 1

    print(f"Done! Valid samples saved: {valid_samples}")


if __name__ == "__main__":
    process_dataset()