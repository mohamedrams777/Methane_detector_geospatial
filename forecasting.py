"""
forecasting.py

7-Day Forward Methane Hotspot Progression Forecasting.

Models microbial methanogenesis kinetics, barometric pressure fluctuations,
and historical satellite multispectral sequence features to predict which
landfill sectors will transition into high-emission hotspot zones over the
upcoming week (Day +1 to Day +7).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

from model import build_spatiotemporal_forecaster


def train_spatiotemporal_forecaster(
    data_path: str = "data/processed/temporal_forecasting_data.npz",
    models_dir: str = "models/forecaster",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    print("\n=======================================================")
    print("  Starting 7-Day Spatio-Temporal Hotspot Forecaster")
    print("=======================================================\n")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Temporal dataset not found: {data_path}. Run generate_sample_data.py first.")

    data = np.load(data_path)
    patch_seq = data["patch_seq"]          # (N, seq_len, 32, 32, 7)
    meteo_forecast = data["meteo_forecast"] # (N, 7, 5) [Temp, Press, Wind_U, Wind_V, Soil_M]
    target_prog = data["target_progression"]# (N, 7)

    n_samples = len(patch_seq)
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.15)

    train_patches, val_patches, test_patches = patch_seq[:n_train], patch_seq[n_train:n_train+n_val], patch_seq[n_train+n_val:]
    train_meteo, val_meteo, test_meteo = meteo_forecast[:n_train], meteo_forecast[n_train:n_train+n_val], meteo_forecast[n_train+n_val:]
    train_target, val_target, test_target = target_prog[:n_train], target_prog[n_train:n_train+n_val], target_prog[n_train+n_val:]

    print(f"Data Split: Train={len(train_patches)}, Val={len(val_patches)}, Test={len(test_patches)}")

    # Instantiate Forecaster
    model = build_spatiotemporal_forecaster(
        seq_length=patch_seq.shape[1],
        spatial_shape=(32, 32, 7),
        meteo_dim=meteo_forecast.shape[2],
        forecast_horizon=7,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
        ],
    )

    save_path = os.path.join(models_dir, "forecaster.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(save_path, monitor="val_roc_auc", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_roc_auc", mode="max", patience=8, restore_best_weights=True),
    ]

    history = model.fit(
        x=[train_patches, train_meteo],
        y=train_target,
        validation_data=([val_patches, val_meteo], val_target),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate Day-by-Day Forecast Accuracy on Held-Out Test Set
    test_preds = model.predict([test_patches, test_meteo])

    print("\n--- 7-Day Forward Hotspot Prediction Performance (Test Split) ---")
    day_metrics = []
    for day in range(7):
        y_t = test_target[:, day]
        y_p = test_preds[:, day]
        roc = roc_auc_score(y_t, y_p)
        pr = average_precision_score(y_t, y_p)
        acc = np.mean((y_p >= 0.5) == y_t)
        day_metrics.append({
            "forecast_day": f"Day +{day+1}",
            "roc_auc": roc,
            "pr_auc": pr,
            "accuracy": acc,
        })
        print(f"  Day +{day+1} Forecast -> ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Acc: {acc:.4f}")

    metrics_df = pd.DataFrame(day_metrics)
    metrics_df.to_csv(os.path.join(models_dir, "day_by_day_forecast_metrics.csv"), index=False)

    # Plot 7-Day Performance Decay Curve
    plt.figure(figsize=(10, 5))
    days = [f"Day +{i+1}" for i in range(7)]
    plt.plot(days, metrics_df["roc_auc"], marker="o", lw=2.5, color="#1f77b4", label="ROC-AUC Score")
    plt.plot(days, metrics_df["pr_auc"], marker="s", lw=2.5, color="#ff7f0e", label="PR-AUC Score")
    plt.title("7-Day Forward Methane Hotspot Forecasting Reliability Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Forecast Horizon", fontsize=12)
    plt.ylabel("Performance Metric", fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    forecast_fig_path = "outputs/figures/7day_forecast_progression.png"
    plt.savefig(forecast_fig_path, dpi=300)
    plt.close()
    print(f"Saved 7-Day Forecast reliability curve to {forecast_fig_path}")

    # Generate sample weekly sector forecast report for landfill operators
    sectors = [
        {"sector_id": "Sector-A (Active Tipping Face)", "current_status": "High Hotspot", "day1_7_risk": test_preds[0].tolist()},
        {"sector_id": "Sector-B (Capped Waste Cell)", "current_status": "Low Hotspot", "day1_7_risk": test_preds[1].tolist()},
        {"sector_id": "Sector-C (Leachate Aeration Pond)", "current_status": "Moderate Hotspot", "day1_7_risk": test_preds[2].tolist()},
        {"sector_id": "Sector-D (Composting & Green Waste)", "current_status": "Low Hotspot", "day1_7_risk": test_preds[3].tolist()},
    ]
    with open("outputs/predictions/weekly_landfill_sector_forecast.json", "w") as f:
        json.dump(sectors, f, indent=2)
    print("Saved sector-level actionable forecast to outputs/predictions/weekly_landfill_sector_forecast.json")


if __name__ == "__main__":
    train_spatiotemporal_forecaster()
