import os

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import json
import torch
import threading
import queue
import numpy as np
import time
from collections import deque
import torch.nn as nn
import subprocess
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "best_model_mediapipe11.pth"
MAPPING_PATH = "class_mapping_mediapipe11.json"
HAND_TASK_PATH = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.90
BUFFER_SIZE = 12
COOLDOWN_SECONDS = 2.0


class LandmarkNN(nn.Module):
    def __init__(self, input_size=126, num_classes=10):
        super(LandmarkNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x): return self.net(x)


class ISLEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_mapping = {}
        self.tts_queue = queue.Queue()
        self.prediction_buffer = deque(maxlen=BUFFER_SIZE)
        self.last_spoken_word = None
        self.last_spoken_time = 0
        self.current_sign = "---"
        self.current_confidence = 0.0
        self.history = ""
        self.is_running = True
        self.camera_active = False
        self.cap = None

        # FPS Calculation
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0

        blank = np.zeros((480, 640, 3), np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank)
        self.blank_frame = jpeg.tobytes()

        threading.Thread(target=self.tts_worker, daemon=True).start()
        self.initialize_engine()

    def tts_worker(self):
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=1.0)
                ps_script = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
                subprocess.Popen(['powershell', '-Command', ps_script], creationflags=subprocess.CREATE_NO_WINDOW)
                self.tts_queue.task_done()
            except:
                pass

    def speak_text(self, text):
        self.tts_queue.put(text)

    def initialize_engine(self):
        try:
            with open(MAPPING_PATH, 'r') as f:
                self.class_mapping = {int(k): v for k, v in json.load(f).items()}
            self.model = LandmarkNN(input_size=126, num_classes=len(self.class_mapping))
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device).eval()
            if self.device.type == "cuda": self.model.half()
        except Exception as e:
            print(f"Model Init Error: {e}")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_TASK_PATH),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_active = True
                return True
        return False

    def stop_camera(self):
        self.camera_active = False
        if self.cap: self.cap.release(); self.cap = None
        return True

    def get_frame(self):
        if not self.camera_active or not self.cap: return self.blank_frame

        # FPS Logic
        self.fps_counter += 1
        if (time.time() - self.fps_start_time) > 1:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()

        ret, frame = self.cap.read()
        if not ret: return self.blank_frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))

        if results.hand_landmarks:
            hands_list = []
            for hand_lms in results.hand_landmarks: hands_list.append((hand_lms[0].x, hand_lms))
            hands_list.sort(key=lambda x: x[0])
            left_data, right_data = [0.0] * 63, [0.0] * 63

            for i, (_, hand_lms) in enumerate(hands_list):
                if i >= 2: break
                base = hand_lms[0]
                max_dist = max(1e-6,
                               max([((lm.x - base.x) ** 2 + (lm.y - base.y) ** 2 + (lm.z - base.z) ** 2) ** 0.5 for lm
                                    in hand_lms]))
                temp = []
                for lm in hand_lms: temp.extend(
                    [(lm.x - base.x) / max_dist, (lm.y - base.y) / max_dist, (lm.z - base.z) / max_dist])
                if i == 0:
                    left_data = temp
                else:
                    right_data = temp

            input_t = torch.tensor(left_data + right_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.device.type == "cuda": input_t = input_t.half()
            with torch.no_grad():
                out = self.model(input_t)
                prob, pred = torch.max(torch.softmax(out, dim=1), 1)
                self.prediction_buffer.append((pred.item(), prob.item()))

            if len(self.prediction_buffer) == BUFFER_SIZE:
                classes = [p[0] for p in self.prediction_buffer]
                probs = [p[1] for p in self.prediction_buffer]
                most_common = max(set(classes), key=classes.count)
                avg_conf = sum(probs) / BUFFER_SIZE

                if classes.count(most_common) >= (BUFFER_SIZE * 0.7) and avg_conf >= CONFIDENCE_THRESHOLD:
                    name = self.class_mapping.get(most_common, "---")
                    self.current_sign = name
                    self.current_confidence = avg_conf
                    if name != "---" and name != self.last_spoken_word:
                        if (time.time() - self.last_spoken_time) > COOLDOWN_SECONDS:
                            self.speak_text(name)
                            self.history = (self.history + " " + name).strip()
                            self.last_spoken_word = name
                            self.last_spoken_time = time.time()
                else:
                    self.current_confidence = avg_conf

        frame = cv2.flip(frame, 1)
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_stats(self):
        return {"sign": self.current_sign, "confidence": round(float(self.current_confidence), 2),
                "history": self.history, "camera_active": self.camera_active, "fps": self.current_fps}

    def release(self):
        self.is_running = False
        self.stop_camera()