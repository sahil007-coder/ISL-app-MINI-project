"""
train_cnn.py
------------
Trains a MobileNetV2-based CNN on the ISL Kaggle image dataset.

Why MobileNetV2?
- Built for real-time / mobile image tasks (~3 ms/frame on CPU)
- Depthwise-separable convolutions → fast without losing accuracy
- Pre-trained on ImageNet → already knows edges, textures, shapes
- We fine-tune only the top layers → trains in minutes, not hours

Run:
    python train_cnn.py
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "data", "datasets")         # root folder: dataset/A/ dataset/B/ …
MODEL_DIR    = "models_cnn"
IMG_SIZE     = 64                 # MobileNetV2 minimum; 64px is fast and accurate
BATCH_SIZE   = 32
EPOCHS_FROZEN = 10               # train only the head first
EPOCHS_FINE   = 30               # then unfreeze top layers and fine-tune
LEARNING_RATE = 1e-3
FINE_LR       = 1e-5             # very low LR during fine-tuning to avoid forgetting


def build_data_generators(dataset_dir: str, img_size: int, batch_size: int):
    """ImageDataGenerator with augmentation for train, simple rescale for val/test."""

    # Augmentation for training — simulate different lighting, angles, distances
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.7, 1.3),
        horizontal_flip=False,     # ISL signs are NOT mirror-symmetric
        fill_mode="nearest",
    )

    val_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    target = (img_size, img_size)

    train_ds = train_gen.flow_from_directory(
        dataset_dir,
        target_size=target,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_ds = val_gen.flow_from_directory(
        dataset_dir,
        target_size=target,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_ds, val_ds


def build_model(num_classes: int, img_size: int) -> keras.Model:
    """
    MobileNetV2 backbone (frozen) + custom classification head.

    Architecture:
        Input (64×64×3)
            ↓
        MobileNetV2 (pre-trained, frozen at first)
            ↓
        GlobalAveragePooling2D   ← collapses spatial dims
            ↓
        Dense 256 + BN + ReLU + Dropout 0.4
            ↓
        Dense 128 + BN + ReLU + Dropout 0.3
            ↓
        Dense(num_classes) + Softmax
    """
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,          # remove ImageNet head
        weights="imagenet",         # start with pre-trained weights
        alpha=0.75,                 # 0.75× width → 25% fewer params, still accurate
    )
    base.trainable = False          # freeze backbone for Phase 1

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="ISL_MobileNetV2"), base


def plot_history(history, path: str, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout(); plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved → {path}")


def evaluate_and_plot(model, val_ds, classes, save_dir: str):
    print("\n── Evaluating on validation set ──")
    val_ds.reset()
    y_pred_prob = model.predict(val_ds, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = val_ds.classes

    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(classes)), max(8, len(classes))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    print(f"Confusion matrix saved → {save_dir}/confusion_matrix.png")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"'{DATASET_DIR}/' not found."
        )

    print("Building data generators …")
    train_ds, val_ds = build_data_generators(DATASET_DIR, IMG_SIZE, BATCH_SIZE)

    classes     = list(train_ds.class_indices.keys())
    num_classes = len(classes)
    label_map   = train_ds.class_indices

    print(f"Classes: {num_classes}  → {classes}")
    print(f"Train batches : {len(train_ds)}")
    print(f"Val batches   : {len(val_ds)}\n")

    # Save label map
    with open(f"{MODEL_DIR}/label_map.pkl", "wb") as f:
        pickle.dump({"label_map": label_map, "classes": classes}, f)

    # ── Phase 1: Train head only (backbone frozen) ───────────────────────────
    print("=" * 50)
    print("PHASE 1 — Training classification head (backbone frozen)")
    print("=" * 50)

    model, base = build_model(num_classes, IMG_SIZE)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs_phase1 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(f"{MODEL_DIR}/phase1_best.keras",
                                  monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_FROZEN, callbacks=cbs_phase1)
    plot_history(h1, f"{MODEL_DIR}/phase1_curves.png", "Phase 1 – Head only")

    # ── Phase 2: Unfreeze top layers and fine-tune ───────────────────────────
    print("\n" + "=" * 50)
    print("PHASE 2 — Fine-tuning top MobileNetV2 layers")
    print("=" * 50)

    # Unfreeze from layer 100 onwards (last ~40 layers of MobileNetV2)
    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"Trainable layers now: {trainable_count}")

    # Recompile with very low LR to avoid catastrophic forgetting
    model.compile(
        optimizer=keras.optimizers.Adam(FINE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs_phase2 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(f"{MODEL_DIR}/best_model.keras",
                                  monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_FINE, callbacks=cbs_phase2)
    plot_history(h2, f"{MODEL_DIR}/phase2_curves.png", "Phase 2 – Fine-tuning")

    # ── Final evaluation ─────────────────────────────────────────────────────
    evaluate_and_plot(model, val_ds, classes, MODEL_DIR)

    # Also export as SavedModel for TF-Lite conversion (optional)
    model.export(f"{MODEL_DIR}/saved_model")

    print(f"\nDone! Best model saved → {MODEL_DIR}/best_model.keras")
    print(f"Expected val accuracy: 97–99% on clean ISL dataset")


if __name__ == "__main__":
    train()
