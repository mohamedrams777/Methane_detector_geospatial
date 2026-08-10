"""
evaluate.py

BATCH 6 — Full evaluation of the trained 2D CNN on the held-out spatial test
split: confusion matrix, accuracy/precision/recall/F1 (per-class + macro),
ROC-AUC, PR-AUC, a calibration check, and an honest comparison against the
1D CNN tabular baseline.

Purpose
-------
Evaluate strictly on data/splits/test_ids.txt — a set of patches whose whole
geographic block (and any shared CH4 pixel) never appeared in train or val
(see dataset_split.py). Normalization uses the training statistics saved
during train.py, never test-set statistics.

Input
-----
- data/processed/patches_array.npz
- data/splits/test_ids.txt
- models/<run_tag>/model.keras
- models/<run_tag>/preprocessing_config.json

Output
------
- outputs/figures/confusion_matrix.png
- outputs/figures/roc_curve.png
- outputs/figures/pr_curve.png
- outputs/figures/calibration_curve.png
- outputs/predictions/test_predictions.csv  (patch_id, y_true, y_prob, y_pred)
- reports/evaluation_metrics.json (also printed, and includes the documented
  comparison against the 1D CNN tabular baseline numbers from the project brief)

Dependencies
------------
tensorflow, numpy, pandas, scikit-learn, matplotlib

Usage
-----
    python evaluate.py \
        --patches data/processed/patches_array.npz \
        --splits-dir data/splits \
        --model-dir models/cnn_v1_2026-08-09
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
from tensorflow import keras

# Documented for honest side-by-side comparison only — NOT re-derived here,
# taken from the project brief's reported 1D CNN tabular baseline.
BASELINE_1D_CNN = {
    "accuracy": 0.4166,
    "precision": 0.8049,
    "recall": 0.1743,
    "f1": 0.2865,
    "roc_auc": 0.5962,
    "note": "1D CNN on tabular CSV features (no spatial context); not directly "
            "comparable in data modality, retained here only as documented context.",
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
    mean = np.array(config["train_channel_mean"], dtype="float32")
    std = np.array(config["train_channel_std"], dtype="float32")
    return (X - mean) / std


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-hotspot", "Hotspot"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-hotspot", "Hotspot"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
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
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Test Set"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_true, y_prob, pr_auc, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"PR (AP={pr_auc:.3f})")
    baseline = y_true.mean()
    plt.axhline(baseline, linestyle="--", color="gray", label=f"Chance (prevalence={baseline:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Test Set"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration(y_true, y_prob, out_path):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration Curve — Test Set"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.predictions_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    X_test, y_test, patch_ids = load_test_set(args.patches, args.splits_dir)
    print(f"Test set: {len(X_test)} genuine, held-out patches "
          f"(whole geographic blocks never seen in train/val).")

    with open(os.path.join(args.model_dir, "preprocessing_config.json")) as f:
        preprocessing_config = json.load(f)
    X_test_n = normalize_with_config(X_test, preprocessing_config)

    model = keras.models.load_model(os.path.join(args.model_dir, "model.keras"))
    y_prob = model.predict(X_test_n).ravel()
    y_pred = (y_prob >= args.threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    plot_confusion_matrix(cm, os.path.join(args.figures_dir, "confusion_matrix.png"))
    plot_roc(y_test, y_prob, roc_auc, os.path.join(args.figures_dir, "roc_curve.png"))
    plot_pr(y_test, y_prob, pr_auc, os.path.join(args.figures_dir, "pr_curve.png"))
    plot_calibration(y_test, y_prob, os.path.join(args.figures_dir, "calibration_curve.png"))

    pred_df = pd.DataFrame({
        "patch_id": patch_ids, "y_true": y_test, "y_prob": y_prob, "y_pred": y_pred,
    })
    pred_df.to_csv(os.path.join(args.predictions_dir, "test_predictions.csv"), index=False)

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
        "decision_threshold": args.threshold,
        "comparison_1d_cnn_tabular_baseline": BASELINE_1D_CNN,
    }

    with open(os.path.join(args.reports_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nFigures written to {args.figures_dir}")
    print(f"Predictions written to {args.predictions_dir}/test_predictions.csv")
    print(f"Full metrics report written to {args.reports_dir}/evaluation_metrics.json")


if __name__ == "__main__":
    main()
