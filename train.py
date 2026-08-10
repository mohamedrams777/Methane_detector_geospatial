"""
train.py

BATCH 5 — Load the split, genuine patch dataset, normalize using train-only
statistics, train the 2D CNN, and save the model plus its preprocessing/label
configuration.

Purpose
-------
Trains build_custom_cnn() (model.py) on the leakage-safe splits produced by
dataset_split.py. Per-channel normalization statistics are computed ONLY from
the training split and then applied to val/test, to avoid statistical
leakage. Saves everything inference.py will need to exactly reproduce
preprocessing later.

Input
-----
- data/processed/patches_array.npz  (X, y, patch_ids from preprocessing.py)
- data/splits/{train,val,test}_ids.txt  (from dataset_split.py)

Output
------
- models/cnn_v1_<date>/model.keras
- models/cnn_v1_<date>/preprocessing_config.json  (per-channel mean/std, band order, patch size)
- models/cnn_v1_<date>/label_threshold.json       (labeling method + parameters used)
- models/cnn_v1_<date>/training_history.csv
- outputs/figures/training_curves.png

Dependencies
------------
tensorflow, numpy, pandas, scikit-learn, matplotlib

Usage
-----
    python train.py \
        --patches data/processed/patches_array.npz \
        --splits-dir data/splits \
        --models-dir models \
        --epochs 150 --batch-size 32
"""

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from model import build_custom_cnn

BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]


def load_split_arrays(patches_path: str, splits_dir: str):
    data = np.load(patches_path, allow_pickle=True)
    X, y, patch_ids = data["X"], data["y"], data["patch_ids"].astype(str)
    id_to_idx = {pid: i for i, pid in enumerate(patch_ids)}

    splits = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(splits_dir, f"{split}_ids.txt")) as f:
            ids = [line.strip() for line in f if line.strip()]
        idxs = [id_to_idx[i] for i in ids if i in id_to_idx]
        missing = len(ids) - len(idxs)
        if missing:
            print(f"Warning: {missing} {split} ids not found in patches_array.npz (dropped).")
        splits[split] = (X[idxs], y[idxs])

    return splits


def compute_train_stats(X_train: np.ndarray):
    """Per-channel mean/std computed ONLY from the training split."""
    mean = X_train.mean(axis=(0, 1, 2))
    std = X_train.std(axis=(0, 1, 2))
    std[std == 0] = 1e-6
    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def augment(image, label):
    """Label-preserving geometric augmentation: random flips/90-degree rotations."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    return image, label


def make_dataset(X, y, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weight(y_train: np.ndarray) -> dict:
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    total = n_pos + n_neg
    return {0: total / (2 * n_neg), 1: total / (2 * n_pos)}


def main():
    parser = argparse.ArgumentParser(description="Train the 2D CNN on genuine methane-hotspot patches.")
    parser.add_argument("--patches", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--run-tag", default=None, help="Override the date-based run folder name")
    args = parser.parse_args()

    splits = load_split_arrays(args.patches, args.splits_dir)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    mean, std = compute_train_stats(X_train)
    X_train_n = normalize(X_train, mean, std)
    X_val_n = normalize(X_val, mean, std)
    X_test_n = normalize(X_test, mean, std)  # normalized here for convenience; evaluate.py re-derives if run standalone

    train_ds = make_dataset(X_train_n, y_train, args.batch_size, training=True)
    val_ds = make_dataset(X_val_n, y_val, args.batch_size, training=False)

    model = build_custom_cnn(input_shape=X_train.shape[1:])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy",
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )

    run_tag = args.run_tag or f"cnn_v1_{date.today().isoformat()}"
    run_dir = os.path.join(args.models_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=args.patience,
                                       restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(run_dir, "model.keras"),
                                         monitor="val_auc", mode="max", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    class_weight = compute_class_weight(y_train)
    print(f"Class weights: {class_weight}")

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=callbacks, class_weight=class_weight,
    )

    # Save preprocessing config so inference.py reproduces normalization exactly.
    preprocessing_config = {
        "band_order": BAND_ORDER,
        "patch_size": int(X_train.shape[1]),
        "reflectance_scale_applied": True,
        "train_channel_mean": mean.tolist(),
        "train_channel_std": std.tolist(),
    }
    with open(os.path.join(run_dir, "preprocessing_config.json"), "w") as f:
        json.dump(preprocessing_config, f, indent=2)

    label_threshold = {
        "primary_label": "label_enhancement",
        "method": "local_background_subtraction",
        "background_radius_m": 25000,
        "background_date_window_days": 30,
        "enhancement_std_multiplier": 1.5,
        "reference_only_method": "global_percentile_75",
    }
    with open(os.path.join(run_dir, "label_threshold.json"), "w") as f:
        json.dump(label_threshold, f, indent=2)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

    os.makedirs("outputs/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_df["loss"], label="train")
    axes[0].plot(hist_df["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(hist_df["auc"], label="train")
    axes[1].plot(hist_df["val_auc"], label="val")
    axes[1].set_title("AUC"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/training_curves.png", dpi=150)
    print(f"\nModel and configs saved to {run_dir}")
    print("Run evaluate.py next on the held-out test split.")


if __name__ == "__main__":
    main()
