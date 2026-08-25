"""
evaluate.py

Full evaluation of the trained model on the held-out spatial test split:
confusion matrix, accuracy/precision/recall/F1, ROC-AUC, PR-AUC, calibration check,
and honest comparison against the 1D CNN tabular baseline.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow import keras

from model import SqueezeAndExcitationBlock, SpatialAttentionBlock, build_multispectral_resnet50, build_custom_cnn

BASELINE_1D_CNN = {
    "accuracy": 0.4166,
    "precision": 0.8049,
    "recall": 0.1743,
    "f1": 0.2865,
    "roc_auc": 0.5962,
    "note": "1D CNN on tabular CSV features (no spatial context); retained here for baseline comparison.",
}


def load_test_set(patches_path: str, splits_dir: str):
    data = np.load(patches_path, allow_pickle=True)
    X, y, patch_ids = data["X"], data["y"], data["patch_ids"].astype(str)
    id_to_idx = {pid: i for i, pid in enumerate(patch_ids)}

    with open(os.path.join(splits_dir, "test_ids.txt")) as f:
        ids = [line.strip() for line in f if line.strip()]
    idxs = [id_to_idx[i] for i in ids if i in id_to_idx]

    return X[idxs], y[idxs], patch_ids[idxs]


def normalize_with_config(X: np.ndarray, config: dict) -> np.ndarray:
    mean_key = "mean" if "mean" in config else "train_channel_mean"
    std_key = "std" if "std" in config else "train_channel_std"
    mean = np.array(config[mean_key], dtype="float32").reshape(1, 1, 1, -1)
    std = np.array(config[std_key], dtype="float32").reshape(1, 1, 1, -1)
    return (X - mean) / std


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-hotspot", "Hotspot"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-hotspot", "Hotspot"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_prob, roc_auc, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_true, y_prob, pr_auc, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AP={pr_auc:.3f})")
    baseline = y_true.mean()
    plt.axhline(baseline, linestyle="--", color="gray", label=f"Chance (prevalence={baseline:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration(y_true, y_prob, out_path):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", lw=2, color="#2ca02c", label="Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve — Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_evaluation(
    patches_path: str,
    splits_dir: str,
    model_dir: str,
    figures_dir: str = "outputs/figures",
    predictions_dir: str = "outputs/predictions",
    reports_dir: str = "reports",
    threshold: float = 0.5,
):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    X_test, y_test, patch_ids = load_test_set(patches_path, splits_dir)
    print(f"Test set: {len(X_test)} genuine, held-out patches (leakage-safe spatial blocks).")

    with open(os.path.join(model_dir, "preprocessing_config.json")) as f:
        preprocessing_config = json.load(f)
    X_test_n = normalize_with_config(X_test, preprocessing_config)

    model_file = os.path.join(model_dir, "model.keras")
    custom_objects = {
        "SqueezeAndExcitationBlock": SqueezeAndExcitationBlock,
        "SpatialAttentionBlock": SpatialAttentionBlock,
    }

    try:
        model = keras.models.load_model(model_file, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Note: Standard load_model had notice ({e}), loading via architecture builder + weights...")
        if "ResNet50" in preprocessing_config.get("backbone", ""):
            model = build_multispectral_resnet50(input_shape=(32, 32, 7))
        else:
            model = build_custom_cnn(input_shape=(32, 32, 7))
        model.load_weights(model_file)

    preds = model.predict(X_test_n)
    if isinstance(preds, dict):
        y_prob = preds["hotspot_prob"].ravel()
    elif isinstance(preds, list):
        y_prob = preds[0].ravel()
    else:
        y_prob = preds.ravel()

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    plot_confusion_matrix(cm, os.path.join(figures_dir, "confusion_matrix.png"))
    plot_roc(y_test, y_prob, roc_auc, os.path.join(figures_dir, "roc_curve.png"))
    plot_pr(y_test, y_prob, pr_auc, os.path.join(figures_dir, "pr_curve.png"))
    plot_calibration(y_test, y_prob, os.path.join(figures_dir, "calibration_curve.png"))

    pred_df = pd.DataFrame({
        "patch_id": patch_ids, "y_true": y_test, "y_prob": y_prob, "y_pred": y_pred,
    })
    pred_df.to_csv(os.path.join(predictions_dir, "test_predictions.csv"), index=False)

    metrics = {
        "n_test_patches": int(len(X_test)),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(acc),
        "precision_per_class": {"non_hotspot": float(precision[0]), "hotspot": float(precision[1])},
        "recall_per_class": {"non_hotspot": float(recall[0]), "hotspot": float(recall[1])},
        "f1_per_class": {"non_hotspot": float(f1[0]), "hotspot": float(f1[1])},
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "decision_threshold": threshold,
        "comparison_1d_cnn_tabular_baseline": BASELINE_1D_CNN,
    }

    with open(os.path.join(reports_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nFigures written to {figures_dir}")
    print(f"Predictions written to {predictions_dir}/test_predictions.csv")
    print(f"Full metrics report written to {reports_dir}/evaluation_metrics.json")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained 2D CNN on the held-out test split.")
    parser.add_argument("--patches", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_evaluation(
        patches_path=args.patches,
        splits_dir=args.splits_dir,
        model_dir=args.model_dir,
        figures_dir=args.figures_dir,
        predictions_dir=args.predictions_dir,
        reports_dir=args.reports_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
