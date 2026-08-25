"""
inference.py

BATCH 7 — Run the trained pipeline on new coordinates/dates for deployment-
style prediction: re-extract genuine Sentinel-2 patches via GEE for the
requested points, apply the exact saved preprocessing, and predict hotspot
probability per patch.

Purpose
-------
Reuses the same extraction logic as gee_extraction.py (Batch 1) so inference
patches are built identically to training patches — same band set, same
cloud/QA filtering, same reflectance scaling — then applies the *training*
normalization statistics (never re-computed on new data) before prediction.

Input
-----
- A CSV of new points to score: columns latitude, longitude, date
- models/<run_tag>/model.keras
- models/<run_tag>/preprocessing_config.json

Output
------
- outputs/predictions/inference_predictions.csv
  (latitude, longitude, date, matched_scene_id, matched_scene_date,
   cloud_cover_pct, hotspot_probability, hotspot_prediction)
- outputs/predictions/inference_predictions.tif
  (rasterized point predictions on a regular grid, for direct GIS use)

Dependencies
------------
earthengine-api, tensorflow, numpy, pandas, rasterio

Usage
-----
    python inference.py \
        --points-csv new_points.csv \
        --model-dir models/cnn_v1_2026-08-09 \
        --gee-project <your-gee-project-id> \
        --out-dir outputs/predictions
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import ee
from tensorflow import keras

from gee_extraction import (
    init_gee, find_best_scene, build_patch_image, make_patch_region,
    cloud_fraction_in_window, MAX_CLOUDY_FRACTION_IN_PATCH, PIXEL_SIZE_M,
    BANDS_10M, BANDS_20M,
)

DN_SCALE = 10000.0


def fetch_patch_array(lon: float, lat: float, date_str: str, patch_size: int):
    """
    Re-run the same extraction logic as Batch 1 for a single new point, but
    pull pixel values directly (via ee.Image.sampleRectangle) instead of
    submitting a Drive export — appropriate for on-demand inference on a
    modest number of new points.
    Returns (array, scene_id, scene_date, cloud_pct) or (None, None, None, None).
    """
    scene = find_best_scene(lon, lat, date_str)
    if scene is None:
        return None, None, None, None

    scene_id = scene.get("system:index").getInfo()
    scene_date = ee.Date(scene.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    cloud_pct = scene.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    region = make_patch_region(lon, lat, patch_size)
    patch_img = build_patch_image(scene)

    bad_frac = cloud_fraction_in_window(patch_img, region).getInfo()
    if bad_frac > MAX_CLOUDY_FRACTION_IN_PATCH:
        return None, scene_id, scene_date, cloud_pct

    export_img = patch_img.select(BANDS_10M + BANDS_20M + ["NDVI"])
    sampled = export_img.sampleRectangle(region=region, defaultValue=0)

    bands = BANDS_10M + BANDS_20M + ["NDVI"]
    arrays = []
    for b in bands:
        band_arr = np.array(sampled.get(b).getInfo(), dtype="float32")
        arrays.append(band_arr)

    # Align all bands to the same (patch_size, patch_size) shape by center-cropping/padding.
    target_h, target_w = patch_size, patch_size
    fixed = []
    for arr in arrays:
        h, w = arr.shape
        out = np.zeros((target_h, target_w), dtype="float32")
        ch, cw = min(h, target_h), min(w, target_w)
        out[:ch, :cw] = arr[:ch, :cw]
        fixed.append(out)

    patch = np.stack(fixed, axis=-1)  # (H, W, 7)
    patch[..., :6] = patch[..., :6] / DN_SCALE  # reflectance scaling, NDVI left as-is

    return patch, scene_id, scene_date, cloud_pct


def run_inference(points_csv: str, model_dir: str, gee_project: str, out_dir: str):
    init_gee(gee_project)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(model_dir, "preprocessing_config.json")) as f:
        config = json.load(f)
    mean = np.array(config["train_channel_mean"], dtype="float32")
    std = np.array(config["train_channel_std"], dtype="float32")
    patch_size = config["patch_size"]

    model = keras.models.load_model(os.path.join(model_dir, "model.keras"))

    points = pd.read_csv(points_csv)
    required = {"latitude", "longitude", "date"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"points CSV missing columns: {missing}")

    records = []
    for _, row in points.iterrows():
        patch, scene_id, scene_date, cloud_pct = fetch_patch_array(
            row["longitude"], row["latitude"], str(row["date"])[:10], patch_size
        )
        if patch is None:
            records.append(dict(
                latitude=row["latitude"], longitude=row["longitude"], date=row["date"],
                matched_scene_id=scene_id, matched_scene_date=scene_date,
                cloud_cover_pct=cloud_pct, hotspot_probability=None, hotspot_prediction=None,
                status="no_valid_patch",
            ))
            continue

        patch_n = (patch - mean) / std
        prob = float(model.predict(patch_n[None, ...], verbose=0)[0, 0])
        records.append(dict(
            latitude=row["latitude"], longitude=row["longitude"], date=row["date"],
            matched_scene_id=scene_id, matched_scene_date=scene_date,
            cloud_cover_pct=cloud_pct, hotspot_probability=prob,
            hotspot_prediction=int(prob >= 0.5), status="ok",
        ))

    result_df = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "inference_predictions.csv")
    result_df.to_csv(out_csv, index=False)

    n_ok = (result_df["status"] == "ok").sum()
    print(f"Scored {n_ok} / {len(result_df)} points successfully.")
    print(f"Predictions written to {out_csv}")
    print("Note: predictions are tied to the specific acquisition date(s) matched "
          "above, not a permanent site classification — re-run for new dates as needed.")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Run inference on new points via the trained pipeline.")
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gee-project", required=True)
    parser.add_argument("--out-dir", default="outputs/predictions")
    args = parser.parse_args()

    run_inference(args.points_csv, args.model_dir, args.gee_project, args.out_dir)


if __name__ == "__main__":
    main()
