"""
gee_extraction.py

BATCH 1 — Genuine Sentinel-2 patch extraction from Google Earth Engine.

Purpose
-------
For each (lat, lon, date) sample in the original GEE-exported CSV, re-query
Earth Engine directly (not the CSV band values) to pull a genuine 32x32 pixel
Sentinel-2 neighborhood, filter it for cloud/QA quality, and export it as a
GeoTIFF to Google Drive. Every exported patch is traceable to a real Sentinel-2
scene ID and acquisition date — nothing here is fabricated or duplicated.

Input
-----
- data/raw/original_gee_export.csv  (must contain: latitude, longitude, date)

Output
------
- GeoTIFF patches written to a Google Drive folder (via GEE batch export)
- data/raw/extraction_log.csv : one row per exported patch with
  patch_id, latitude, longitude, requested_date, matched_scene_id,
  matched_scene_date, cloud_cover_pct, export_task_id, status

Dependencies
------------
earthengine-api, pandas, numpy, tqdm

Usage
-----
    python gee_extraction.py --csv data/raw/original_gee_export.csv \
        --drive-folder methane_patches --patch-size 32 \
        --gee-project <your-gee-project-id>

Notes
-----
- This script only SUBMITS export tasks; GEE exports run server-side and take
  time. Task completion must be polled separately (see check_export_status()
  and notebooks/01_gee_patch_extraction.ipynb).
- Patch center coordinates are preserved exactly; the 32x32 window is built
  at 10m/pixel using the point as the window center.
"""

import argparse
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import ee

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS_10M = ["B2", "B3", "B4", "B8"]
BANDS_20M = ["B11", "B12"]
QA_BAND = "SCL"  # Scene Classification Layer: cloud/shadow/snow flags
CLOUD_SCL_CLASSES = [3, 8, 9, 10, 11]  # shadow, cloud med/high prob, cirrus, snow
MAX_CLOUDY_FRACTION_IN_PATCH = 0.10    # reject patch if >10% of pixels are bad
DATE_WINDOW_DAYS = 5                   # +/- window around the requested date
PIXEL_SIZE_M = 10


def init_gee(project_id: str):
    """Authenticate and initialize Earth Engine for the given project."""
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def find_best_scene(lon: float, lat: float, date_str: str, window_days: int = DATE_WINDOW_DAYS):
    """
    Find the least-cloudy Sentinel-2 SR scene covering (lon, lat) within
    +/- window_days of date_str. Returns an ee.Image or None if no match.
    """
    point = ee.Geometry.Point([lon, lat])
    center_date = ee.Date(date_str)
    start = center_date.advance(-window_days, "day")
    end = center_date.advance(window_days, "day")

    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(point)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = coll.size().getInfo()
    if count == 0:
        return None
    return coll.first()


def build_patch_image(scene: "ee.Image") -> "ee.Image":
    """
    Assemble the 7-channel patch image: B2,B3,B4,B8 (native 10m),
    B11,B12 resampled to 10m, and NDVI computed from B8/B4.
    """
    bands_10m = scene.select(BANDS_10M)
    bands_20m = scene.select(BANDS_20M).resample("bilinear").reproject(
        crs=scene.select("B2").projection(), scale=PIXEL_SIZE_M
    )
    ndvi = scene.normalizedDifference(["B8", "B4"]).rename("NDVI")
    scl = scene.select(QA_BAND)

    return bands_10m.addBands(bands_20m).addBands(ndvi).addBands(scl)


def cloud_fraction_in_window(patch_img: "ee.Image", region: "ee.Geometry") -> float:
    """Compute the fraction of pixels in `region` flagged as cloud/shadow/snow by SCL."""
    scl = patch_img.select(QA_BAND)
    bad_mask = scl.remap(CLOUD_SCL_CLASSES, [1] * len(CLOUD_SCL_CLASSES), 0)
    stats = bad_mask.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=PIXEL_SIZE_M, maxPixels=1e6
    )
    val = stats.get(QA_BAND)
    return ee.Algorithms.If(val, val, 1.0)  # if missing, treat as fully bad


def make_patch_region(lon: float, lat: float, patch_size: int) -> "ee.Geometry":
    """Build a square region of patch_size x patch_size pixels centered on (lon, lat)."""
    half_extent_m = (patch_size / 2.0) * PIXEL_SIZE_M
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(half_extent_m).bounds()


def process_row(row, patch_size: int, drive_folder: str, idx: int):
    """
    Process a single CSV row: find scene, quality-check, submit export task.
    Returns a log dict describing the outcome.
    """
    lat, lon, date_str = row["latitude"], row["longitude"], str(row["date"])[:10]
    patch_id = f"patch_{idx:05d}"

    log = dict(
        patch_id=patch_id, latitude=lat, longitude=lon, requested_date=date_str,
        matched_scene_id=None, matched_scene_date=None, cloud_cover_pct=None,
        export_task_id=None, status="skipped_no_scene",
    )

    scene = find_best_scene(lon, lat, date_str)
    if scene is None:
        return log

    scene_id = scene.get("system:index").getInfo()
    scene_date = ee.Date(scene.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    cloud_pct = scene.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    region = make_patch_region(lon, lat, patch_size)
    patch_img = build_patch_image(scene)

    bad_frac = cloud_fraction_in_window(patch_img, region).getInfo()
    if bad_frac > MAX_CLOUDY_FRACTION_IN_PATCH:
        log.update(matched_scene_id=scene_id, matched_scene_date=scene_date,
                    cloud_cover_pct=cloud_pct, status="rejected_cloudy")
        return log

    export_img = patch_img.select(BANDS_10M + BANDS_20M + ["NDVI"]).clip(region)
    task = ee.batch.Export.image.toDrive(
        image=export_img,
        description=patch_id,
        folder=drive_folder,
        fileNamePrefix=patch_id,
        region=region,
        scale=PIXEL_SIZE_M,
        crs="EPSG:4326",
        maxPixels=1e8,
    )
    task.start()

    log.update(matched_scene_id=scene_id, matched_scene_date=scene_date,
               cloud_cover_pct=cloud_pct, export_task_id=task.id, status="export_submitted")
    return log


def check_export_status(log_df: pd.DataFrame, poll_seconds: int = 30) -> pd.DataFrame:
    """
    Poll GEE for the status of all submitted export tasks and update the log.
    Call this separately after process_row has submitted tasks, since exports
    run asynchronously server-side and can take minutes.
    """
    submitted = log_df[log_df["status"] == "export_submitted"].copy()
    task_ids = submitted["export_task_id"].tolist()

    pending = set(task_ids)
    final_status = {}
    while pending:
        for tid in list(pending):
            task = ee.batch.Task(tid)
            state = task.status()["state"]
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                final_status[tid] = state
                pending.discard(tid)
        if pending:
            time.sleep(poll_seconds)

    log_df.loc[log_df["export_task_id"].isin(final_status.keys()), "status"] = \
        log_df["export_task_id"].map(final_status).fillna(log_df["status"])
    return log_df


def main():
    parser = argparse.ArgumentParser(description="Extract genuine Sentinel-2 patches via GEE.")
    parser.add_argument("--csv", required=True, help="Path to original_gee_export.csv")
    parser.add_argument("--drive-folder", default="methane_patches")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--gee-project", required=True)
    parser.add_argument("--out-log", default="data/raw/extraction_log.csv")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of rows processed (for test runs)")
    args = parser.parse_args()

    init_gee(args.gee_project)

    df = pd.read_csv(args.csv)
    required_cols = {"latitude", "longitude", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if args.limit:
        df = df.head(args.limit)

    logs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Submitting patch exports"):
        logs.append(process_row(row, args.patch_size, args.drive_folder, idx))

    log_df = pd.DataFrame(logs)
    log_df.to_csv(args.out_log, index=False)

    n_submitted = (log_df["status"] == "export_submitted").sum()
    n_rejected = (log_df["status"] == "rejected_cloudy").sum()
    n_no_scene = (log_df["status"] == "skipped_no_scene").sum()
    print(f"Submitted: {n_submitted} | Rejected (cloudy): {n_rejected} | No scene found: {n_no_scene}")
    print(f"Log written to {args.out_log}")
    print("Run check_export_status() separately (or via the notebook) to confirm "
          "Drive downloads have completed before moving to preprocessing.")


if __name__ == "__main__":
    main()
