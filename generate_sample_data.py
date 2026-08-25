"""
generate_sample_data.py

Generates realistic multispectral 32x32x7 patch arrays, temporal time-series,
and leakage-safe spatial splits for the Vellakkal Landfill, Madurai.

This enables immediate, standalone execution, training, and verification of:
1. MultiSpectral-ResNet50
2. Custom 2D CNN
3. 7-Day Spatio-Temporal Forecaster
4. 3D Planar Interactive Map
"""

import os
import numpy as np
import pandas as pd

# Constants for Vellakkal Landfill Region (Madurai, Tamil Nadu)
VELLAKKAL_LAT = 9.8732
VELLAKKAL_LON = 78.0813
BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]


def generate_synthetic_multispectral_dataset(
    num_samples: int = 1200,
    patch_size: int = 32,
    out_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    seed: int = 42,
):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    print(f"Generating {num_samples} multispectral 32x32x7 patches centered on Vellakkal Landfill...")

    X = np.zeros((num_samples, patch_size, patch_size, 7), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int32)
    y_enhancement = np.zeros(num_samples, dtype=np.float32)
    patch_ids = []
    metadata_rows = []

    # Grid of coordinates centered around Vellakkal dumpsite
    lat_offsets = np.random.uniform(-0.06, 0.06, num_samples)
    lon_offsets = np.random.uniform(-0.06, 0.06, num_samples)

    dates = pd.date_range(start="2022-01-01", end="2025-12-31", periods=num_samples)

    for i in range(num_samples):
        pid = f"patch_{i:05d}"
        patch_ids.append(pid)

        lat = VELLAKKAL_LAT + lat_offsets[i]
        lon = VELLAKKAL_LON + lon_offsets[i]

        # Calculate distance to central dumpsite active zone (approx 0,0 offset)
        dist_to_center = np.sqrt(lat_offsets[i]**2 + lon_offsets[i]**2)

        # Hotspot probability increases near the dumpsite active core (< 0.02 deg ~ 2 km)
        is_hotspot_prob = np.clip(1.0 - (dist_to_center / 0.05), 0.1, 0.85)
        is_hotspot = int(np.random.rand() < is_hotspot_prob)
        y[i] = is_hotspot

        # Base regional CH4 background ~1910 ppb, hotspot enhancement 20-95 ppb
        bg_ch4 = 1910.0 + np.random.normal(0, 5.0)
        enhancement = (np.random.uniform(25.0, 90.0) if is_hotspot else np.random.uniform(-10.0, 12.0))
        total_ch4 = bg_ch4 + enhancement
        y_enhancement[i] = max(0.0, enhancement)

        # Generate realistic 7-channel reflectance
        # B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR-1), B12 (SWIR-2), NDVI
        if is_hotspot:
            # Hotspot signatures: Exposed dry bare soil/refuse, low NDVI, elevated SWIR reflectance/absorption
            b2 = np.random.uniform(0.08, 0.16, (patch_size, patch_size))
            b3 = np.random.uniform(0.10, 0.18, (patch_size, patch_size))
            b4 = np.random.uniform(0.12, 0.22, (patch_size, patch_size))
            b8 = np.random.uniform(0.18, 0.28, (patch_size, patch_size))
            b11 = np.random.uniform(0.24, 0.38, (patch_size, patch_size))
            b12 = np.random.uniform(0.20, 0.34, (patch_size, patch_size))
        else:
            # Non-hotspot signatures: Vegetated/soil surround, higher NIR, lower SWIR
            b2 = np.random.uniform(0.04, 0.09, (patch_size, patch_size))
            b3 = np.random.uniform(0.06, 0.12, (patch_size, patch_size))
            b4 = np.random.uniform(0.05, 0.14, (patch_size, patch_size))
            b8 = np.random.uniform(0.25, 0.50, (patch_size, patch_size))
            b11 = np.random.uniform(0.12, 0.22, (patch_size, patch_size))
            b12 = np.random.uniform(0.08, 0.18, (patch_size, patch_size))

        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)

        # Add spatial texture / gradient
        grad_x = np.linspace(-0.02, 0.02, patch_size).reshape(1, -1)
        grad_y = np.linspace(-0.02, 0.02, patch_size).reshape(-1, 1)
        spatial_noise = np.sin(grad_x * 10) + np.cos(grad_y * 10)

        patch_tensor = np.stack([
            b2 + spatial_noise * 0.01,
            b3 + spatial_noise * 0.01,
            b4 + spatial_noise * 0.01,
            b8 + spatial_noise * 0.01,
            b11 + spatial_noise * 0.02,
            b12 + spatial_noise * 0.02,
            ndvi
        ], axis=-1).astype(np.float32)

        X[i] = patch_tensor

        metadata_rows.append({
            "patch_id": pid,
            "latitude": lat,
            "longitude": lon,
            "date": str(dates[i].date()),
            "s5p_ch4_column": total_ch4,
            "ch4_background": bg_ch4,
            "ch4_enhancement": enhancement,
            "label_enhancement": is_hotspot,
            "dist_to_landfill_km": dist_to_center * 111.0,
        })

    # Save array dataset
    npz_path = os.path.join(out_dir, "patches_array.npz")
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        y_enhancement=y_enhancement,
        patch_ids=np.array(patch_ids, dtype=object),
    )
    print(f"Saved patches array to {npz_path} (Shape: {X.shape})")

    # Save metadata
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(out_dir, "metadata_final.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved metadata to {meta_path}")

    # Generate Spatial Block Splits (Leakage-Safe)
    # Block into 4x4 spatial quadrants based on Lat/Lon
    meta_df["block_x"] = pd.qcut(meta_df["longitude"], q=4, labels=False)
    meta_df["block_y"] = pd.qcut(meta_df["latitude"], q=4, labels=False)
    meta_df["spatial_block"] = meta_df["block_x"].astype(str) + "_" + meta_df["block_y"].astype(str)

    unique_blocks = meta_df["spatial_block"].unique()
    np.random.shuffle(unique_blocks)

    n_blocks = len(unique_blocks)
    train_blocks = unique_blocks[: int(n_blocks * 0.70)]
    val_blocks = unique_blocks[int(n_blocks * 0.70) : int(n_blocks * 0.85)]
    test_blocks = unique_blocks[int(n_blocks * 0.85) :]

    train_ids = meta_df[meta_df["spatial_block"].isin(train_blocks)]["patch_id"].tolist()
    val_ids = meta_df[meta_df["spatial_block"].isin(val_blocks)]["patch_id"].tolist()
    test_ids = meta_df[meta_df["spatial_block"].isin(test_blocks)]["patch_id"].tolist()

    meta_df["split"] = "train"
    meta_df.loc[meta_df["patch_id"].isin(val_ids), "split"] = "val"
    meta_df.loc[meta_df["patch_id"].isin(test_ids), "split"] = "test"
    meta_df.to_csv(os.path.join(out_dir, "metadata_with_splits.csv"), index=False)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with open(os.path.join(splits_dir, f"{split_name}_ids.txt"), "w") as f:
            for item in ids:
                f.write(f"{item}\n")

    print(f"Spatial Block Splits created: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Generate Spatio-Temporal Sequence & Meteorological Data for 7-Day Forecasting
    generate_synthetic_forecasting_data(out_dir, num_sequences=300, seq_len=4, forecast_horizon=7)


def generate_synthetic_forecasting_data(out_dir: str, num_sequences: int = 300, seq_len: int = 4, forecast_horizon: int = 7):
    """
    Generates synthetic temporal sequences and meteorological drivers:
    - patch_seq: (N, 4, 32, 32, 7)
    - meteo_forecast: (N, 7, 5) -> [Surface Temp (°C), Pressure (hPa), Wind U (m/s), Wind V (m/s), Soil Moisture (%)]
    - target_progression: (N, 7) -> Binary hotspot presence for next 7 days
    """
    np.random.seed(42)
    patch_seq = np.random.uniform(0.05, 0.45, (num_sequences, seq_len, 32, 32, 7)).astype(np.float32)
    
    # Simulate ERA5 meteorological forecast
    # Temp ~ 28-38°C (Madurai climate), Pressure ~ 1005-1018 hPa, Wind ~ 2-8 m/s, Soil Moisture ~ 10-40%
    temp = np.random.uniform(28.0, 38.0, (num_sequences, forecast_horizon, 1))
    pressure = np.random.uniform(1005.0, 1018.0, (num_sequences, forecast_horizon, 1))
    wind_u = np.random.uniform(-5.0, 5.0, (num_sequences, forecast_horizon, 1))
    wind_v = np.random.uniform(-5.0, 5.0, (num_sequences, forecast_horizon, 1))
    soil_m = np.random.uniform(10.0, 40.0, (num_sequences, forecast_horizon, 1))
    meteo_forecast = np.concatenate([temp, pressure, wind_u, wind_v, soil_m], axis=-1).astype(np.float32)

    # Microbial methanogenesis kinetics:
    # High temp + high moisture + barometric pressure drops induce high hotspot emergence risk
    temp_factor = (temp[:, :, 0] - 28.0) / 10.0
    pressure_drop = np.gradient(pressure[:, :, 0], axis=1)
    risk_score = 0.3 * temp_factor - 0.2 * pressure_drop + 0.01 * soil_m[:, :, 0]
    target_progression = (risk_score > np.median(risk_score)).astype(np.float32)

    seq_path = os.path.join(out_dir, "temporal_forecasting_data.npz")
    np.savez_compressed(
        seq_path,
        patch_seq=patch_seq,
        meteo_forecast=meteo_forecast,
        target_progression=target_progression,
    )
    print(f"Saved temporal forecasting dataset to {seq_path} (Sequences: {num_sequences})")


if __name__ == "__main__":
    generate_synthetic_multispectral_dataset()
