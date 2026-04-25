"""
live_cnn.py
-----------
Real-time ISL interpreter using the trained MobileNetV2 CNN.

Pipeline per frame:
  1. Capture frame from webcam
  2. Crop ROI (center fallback instead of MediaPipe)
  3. Resize ROI to 64×64
  4. CNN inference
  5. Overlay prediction + sentence builder
"""

import pickle
import time
import collections

import cv2
import numpy as np
import tensorflow as tf

# ── Config ─────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models_cnn", "best_model.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models_cnn", "label_map.pkl")

IMG_SIZE             = 64
CONFIDENCE_THRESHOLD = 0.80
STABLE_FRAMES        = 20
CAMERA_INDEX         = 0

# ── Colours (BGR) ──────────────────────────────────────────────────
CLR_ACCENT  = (0, 200, 255)
CLR_GREEN   = (0, 220, 100)
CLR_WHITE   = (255, 255, 255)
CLR_DARK    = (25,  25,  25)

# ── Load model ─────────────────────────────────────────────────────
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "rb") as f:
        meta = pickle.load(f)
    idx_to_class = {v: k for k, v in meta["label_map"].items()}
    return model, idx_to_class


# ── Preprocess ─────────────────────────────────────────────────────
def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = rgb.astype(np.float32) / 255.0
    return norm[np.newaxis]


# ── Center crop (replaces MediaPipe ROI) ───────────────────────────
def get_center_roi(frame: np.ndarray):
    h, w = frame.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    return frame[
        cy - side//2: cy + side//2,
        cx - side//2: cx + side//2
    ]


# ── Main loop ──────────────────────────────────────────────────────
def run():
    print("Loading CNN model …")
    model, idx_to_class = load_model()
    print("Model loaded ✓\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    sentence      = ""
    pred_history  = collections.deque(maxlen=STABLE_FRAMES)
    prev_time     = time.time()

    # Warmup
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)

    print("Starting live feed …\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── ROI (center crop instead of MediaPipe) ─────────────────
        roi = get_center_roi(frame)

        # ── Prediction ─────────────────────────────────────────────
        inp   = preprocess_roi(roi)
        probs = model.predict(inp, verbose=0)[0]

        top        = int(np.argmax(probs))
        confidence = float(probs[top])

        label = None
        if confidence >= CONFIDENCE_THRESHOLD:
            label = idx_to_class[top]

        # ── Stability tracking ─────────────────────────────────────
        pred_history.append(label)
        if len(set(pred_history)) == 1 and label is not None:
            stable_count = len(pred_history)
        else:
            stable_count = 0

        # ── FPS ────────────────────────────────────────────────────
        cur_time  = time.time()
        fps       = 1.0 / (cur_time - prev_time + 1e-9)
        prev_time = cur_time

        # ── UI ─────────────────────────────────────────────────────
        cv2.putText(frame, f"{label} ({confidence:.2f})",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    CLR_GREEN if stable_count >= STABLE_FRAMES else CLR_ACCENT, 2)

        cv2.putText(frame, f"Sentence: {sentence}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    CLR_WHITE, 2)

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    CLR_WHITE, 1)

        cv2.imshow("ISL Interpreter — CNN", frame)

        # ── Key handling ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == 32:
            if stable_count >= STABLE_FRAMES and label:
                sentence += label
                print(f"Added {label!r} → {sentence!r}")
                pred_history.clear()
        elif key == 8 and sentence:
            sentence = sentence[:-1]
        elif key == 13:
            print(f"\n>>> {sentence}\n")
            sentence = ""
        elif key == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    print("Interpreter closed.")


if __name__ == "__main__":
    run()