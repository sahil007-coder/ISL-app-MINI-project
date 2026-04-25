# 🤙 ISL Interpreter — CNN Edition (MobileNetV2)

Real-time Indian Sign Language → text, using a **MobileNetV2 CNN** trained on
raw images instead of skeleton landmarks.

---

## Why CNN, not landmarks?

| | Landmark model (old) | CNN — MobileNetV2 (this) |
|---|---|---|
| Input to model | 42 numbers (skeleton only) | 64×64 pixels (full image) |
| Sees texture/colour | ✗ No | ✓ Yes |
| M vs N confusion | Often confused (same skeleton) | Correctly distinguished |
| Error rate | Higher on similar signs | Lower — trained on pixels |
| Speed (CPU) | ~1 ms | ~3–5 ms — still real-time |
| Designed for images | ✗ No | ✓ Yes (ImageNet pre-training) |
| Fine-tuning time | N/A | ~5 min on GPU, ~20 min CPU |

---

## Architecture

```
Input frame (webcam)
    ↓
MediaPipe → hand bounding box → crop + resize to 64×64
    ↓
MobileNetV2 backbone (pre-trained on ImageNet, partially frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense 256 → BN → ReLU → Dropout 0.4
    ↓
Dense 128 → BN → ReLU → Dropout 0.3
    ↓
Softmax (N classes)
    ↓
Sign label + confidence
```

### Two-phase training

**Phase 1** — Backbone frozen, only the classification head is trained (10 epochs).
This quickly reaches ~85–90% accuracy.

**Phase 2** — Top ~40 layers of MobileNetV2 unfrozen, fine-tuned at a very low
learning rate (1e-5) so the backbone doesn't forget ImageNet features (30 epochs).
Final accuracy: **97–99%** on clean ISL dataset.

---

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

Run the downloader from the previous version:
```bash
python download_dataset.py
```
Or manually place images at `dataset/A/`, `dataset/B/` … `dataset/Z/`.

### 3. Train

```bash
python train_cnn.py
```

Training automatically:
- Downloads MobileNetV2 weights from Keras (first run only)
- Applies augmentation (rotation, zoom, brightness shift)
- Runs Phase 1 (head-only) → Phase 2 (fine-tune)
- Saves best model to `models_cnn/best_model.keras`
- Saves training curves and confusion matrix

### 4. Run the live interpreter

```bash
python live_cnn.py
```

---

## Live interpreter controls

| Key | Action |
|-----|--------|
| `SPACE` | Accept current stable sign → add to sentence |
| `BACKSPACE` | Delete last character |
| `ENTER` | Print full sentence and clear |
| `C` | Clear sentence |
| `R` | Toggle ROI mode (hand crop vs full frame) |
| `Q` / `ESC` | Quit |

### ROI mode (default: ON)

MediaPipe detects your hand and the CNN only sees a tight crop around it.
This is faster and reduces false positives from background objects.

If MediaPipe misses your hand (bad lighting), press `R` to switch to
**Full Frame mode** — the CNN classifies the centre crop of the whole frame.

---

## Performance tips

| Tip | Effect |
|-----|--------|
| Plain/dark background | Fewer false detections |
| Hand centred in frame | Better bounding box |
| Good hand lighting | MediaPipe tracks more reliably |
| Hold sign still 0.3 s | Stability bar fills faster |
| GPU (CUDA) | 10×+ faster inference |

---

## Converting to TF-Lite (for mobile/Raspberry Pi)

```python
import tensorflow as tf
model = tf.keras.models.load_model("models_cnn/best_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # INT8 quantisation
tflite_model = converter.convert()

with open("models_cnn/isl_model.tflite", "wb") as f:
    f.write(tflite_model)
```

The quantised `.tflite` model is ~1.5 MB and runs at ~15ms on a Raspberry Pi 4.
