"""
train_resnet.py

Trains Multi-Spectral ResNet-50 (MS-ResNet50) on leakage-safe spatial splits
for Sentinel-2 + Sentinel-5P Methane Hotspot Detection and Enhancement Regression.

Features:
- 7-channel multispectral input (B2, B3, B4, B8, B11, B12, NDVI)
- Squeeze-and-Excitation (SE) channel attention
- Multi-Task learning: Hotspot Classification (Binary) + Methane Enhancement (Continuous ppb)
- Focal Loss / Weighted BCE for class balance
- Train-only channel normalization to prevent statistical data leakage
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
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from model import build_multispectral_resnet50

BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]


def focal_binary_cross_entropy(gamma=2.0, alpha=0.65):
    """
    Focal Loss for addressing class imbalance by down-weighting easy negative examples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        loss_val = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_val)
    return loss


def load_split_arrays(patches_path: str, splits_dir: str):
    data = np.load(patches_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    y_enh = data["y_enhancement"] if "y_enhancement" in data else np.zeros_like(y, dtype=np.float32)
    patch_ids = data["patch_ids"].astype(str)
    
    id_to_idx = {pid: i for i, pid in enumerate(patch_ids)}

    splits = {}
    for split in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split}_ids.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        idxs = [id_to_idx[i] for i in ids if i in id_to_idx]
        splits[split] = {
            "X": X[idxs],
            "y": y[idxs],
            "y_enh": y_enh[idxs],
            "ids": [patch_ids[i] for i in idxs],
        }
        print(f"Loaded {split} split: {len(idxs)} samples, {int((y[idxs] == 1).sum())} hotspots")

    return splits


def compute_train_normalization(X_train: np.ndarray):
    """
    Compute channel-wise mean and std strictly on training split.
    """
    mean = np.mean(X_train, axis=(0, 1, 2), dtype=np.float64)
    std = np.std(X_train, axis=(0, 1, 2), dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)


def train_ms_resnet(
    patches_path: str = "data/processed/patches_array.npz",
    splits_dir: str = "data/splits",
    models_dir: str = "models",
    epochs: int = 35,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    freeze_backbone: bool = False,
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    run_tag = f"ms_resnet50_{date.today().isoformat()}"
    run_dir = os.path.join(models_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"  Starting Multi-Spectral ResNet-50 Training")
    print(f"  Run Directory: {run_dir}")
    print(f"==========================================\n")

    splits = load_split_arrays(patches_path, splits_dir)

    mean, std = compute_train_normalization(splits["train"]["X"])
    print("Computed Train Channel Statistics:")
    for b, m, s in zip(BAND_ORDER, mean, std):
        print(f"  {b:6s} -> Mean: {m:8.4f}, Std: {s:8.4f}")

    # Apply normalization
    X_train_norm = apply_normalization(splits["train"]["X"], mean, std)
    X_val_norm = apply_normalization(splits["val"]["X"], mean, std)
    X_test_norm = apply_normalization(splits["test"]["X"], mean, std)

    y_train_cls = splits["train"]["y"]
    y_val_cls = splits["val"]["y"]
    y_test_cls = splits["test"]["y"]

    y_train_enh = splits["train"]["y_enh"]
    y_val_enh = splits["val"]["y_enh"]
    y_test_enh = splits["test"]["y_enh"]

    # Class weighting for binary cross entropy
    n_pos = (y_train_cls == 1).sum()
    n_neg = (y_train_cls == 0).sum()
    pos_weight = float(n_neg) / max(1.0, float(n_pos))
    print(f"\nTrain Class Ratio: {n_pos} Pos / {n_neg} Neg (Pos Weight: {pos_weight:.2f})")

    # Build Model
    model = build_multispectral_resnet50(
        input_shape=(32, 32, 7),
        num_classes=1,
        freeze_backbone=freeze_backbone,
        dropout_rate=0.4,
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    losses = {
        "hotspot_prob": focal_binary_cross_entropy(gamma=2.0, alpha=0.65),
        "enhancement_ppb": keras.losses.Huber(delta=5.0),
    }
    loss_weights = {
        "hotspot_prob": 1.0,
        "enhancement_ppb": 0.05,
    }

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "hotspot_prob": [
                keras.metrics.BinaryAccuracy(name="acc"),
                keras.metrics.AUC(name="roc_auc", curve="ROC"),
                keras.metrics.AUC(name="pr_auc", curve="PR"),
            ],
            "enhancement_ppb": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )

    # Callbacks
    model_save_path = os.path.join(run_dir, "model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_hotspot_prob_roc_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_hotspot_prob_roc_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x=X_train_norm,
        y={"hotspot_prob": y_train_cls, "enhancement_ppb": y_train_enh},
        validation_data=(
            X_val_norm,
            {"hotspot_prob": y_val_cls, "enhancement_ppb": y_val_enh},
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save Preprocessing Configuration
    config = {
        "band_order": BAND_ORDER,
        "patch_size": 32,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pos_weight": pos_weight,
        "backbone": "MultiSpectral_ResNet50_with_SE_Attention",
    }
    with open(os.path.join(run_dir, "preprocessing_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save Training History CSV
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

    # Plot & Save Training Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(hist_df["loss"], label="Train Loss", lw=2)
    axes[0].plot(hist_df["val_loss"], label="Val Loss", lw=2)
    axes[0].set_title("Multi-Task Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist_df["hotspot_prob_roc_auc"], label="Train ROC-AUC", lw=2)
    axes[1].plot(hist_df["val_hotspot_prob_roc_auc"], label="Val ROC-AUC", lw=2)
    axes[1].set_title("Hotspot Classification ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(hist_df["hotspot_prob_pr_auc"], label="Train PR-AUC", lw=2)
    axes[2].plot(hist_df["val_hotspot_prob_pr_auc"], label="Val PR-AUC", lw=2)
    axes[2].set_title("Hotspot Classification PR-AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    curve_path = "outputs/figures/ms_resnet50_training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {curve_path}")

    # Evaluate on Held-out Spatial Test Split
    print("\n--- Evaluating MS-ResNet50 on Held-Out Spatial Test Split ---")
    test_preds = model.predict(X_test_norm)
    y_test_probs = test_preds["hotspot_prob"].flatten()
    y_test_preds_binary = (y_test_probs >= 0.5).astype(int)

    test_roc = roc_auc_score(y_test_cls, y_test_probs)
    test_pr = average_precision_score(y_test_cls, y_test_probs)

    print(f"Test ROC-AUC : {test_roc:.4f}")
    print(f"Test PR-AUC  : {test_pr:.4f}")
    print("\nClassification Report (Spatial Test Split):")
    print(classification_report(y_test_cls, y_test_preds_binary, target_names=["Non-Hotspot", "Hotspot"]))

    # Save Test Predictions
    test_pred_df = pd.DataFrame({
        "patch_id": splits["test"]["ids"],
        "y_true": y_test_cls,
        "y_prob": y_test_probs,
        "y_pred": y_test_preds_binary,
        "pred_enhancement_ppb": test_preds["enhancement_ppb"].flatten(),
    })
    test_pred_path = "outputs/predictions/ms_resnet50_test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Saved test predictions to {test_pred_path}")

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", default="data/processed/patches_array.npz")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train_ms_resnet(
        patches_path=args.patches,
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
