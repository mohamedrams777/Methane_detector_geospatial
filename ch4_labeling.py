"""
ch4_labeling.py

BATCH 2 — Associate genuine Sentinel-5P CH4 values with each extracted patch,
and assign hotspot / non-hotspot labels using a background-subtracted
enhancement method (not a bare global percentile).

Purpose
-------
Sentinel-5P TROPOMI CH4 has a native footprint of roughly 7x5.5 km, far
coarser than a 320m Sentinel-2 patch. Many patches will legitimately share
the same underlying CH4 pixel — that is a physical fact of combining these
two sensors and must be preserved and documented, not hidden or "fixed".

This script:
  1. For each successfully exported patch (from extraction_log.csv), finds the
     closest-in-time, QA-passing Sentinel-5P CH4 retrieval covering that point.
  2. Records which S5P pixel (by id/coords) supplied the value, so patches
     sharing a CH4 source are traceable later for leakage-safe splitting.
  3. Computes a local background CH4 level from a wider surrounding area/time
     window (excluding the dumpsite itself) and expresses each patch's CH4 as
     an enhancement over that background.
  4. Assigns hotspot labels from the enhancement (primary method) and, for
     comparison/documentation only, from the legacy global 75th-percentile
     method.

Input
-----
- data/raw/extraction_log.csv   (from gee_extraction.py; status == 'COMPLETED')
- data/raw/original_gee_export.csv  (for the dumpsite bounding context)

Output
------
- data/processed/metadata.csv : one row per patch with
  patch_id, latitude, longitude, patch_date,
  ch4_value, ch4_uncertainty, ch4_qa, ch4_pixel_id, ch4_pixel_date,
  ch4_background, ch4_enhancement,
  label_enhancement (primary), label_percentile75 (reference only)

Dependencies
------------
earthengine-api, pandas, numpy

Usage
-----
    python ch4_labeling.py \
        --extraction-log data/raw/extraction_log.csv \
        --gee-project <your-gee-project-id> \
        --out data/processed/metadata.csv
"""

import argparse
import numpy as np
import pandas as pd
import ee

S5P_COLLECTION = "COPERNICUS/S5P/OFFL/L3_CH4"
CH4_BAND = "CH4_column_volume_mixing_ratio_dry_air"
QA_BAND = "qa_value"
MIN_QA = 0.5                     # TROPOMI-recommended CH4 QA threshold
CH4_DATE_WINDOW_DAYS = 3         # +/- window for matching S2 patch date to S5P retrieval
BACKGROUND_RADIUS_M = 25000      # 25 km ring around the dumpsite for background estimate
BACKGROUND_DATE_WINDOW_DAYS = 30 # wider window for a stable background estimate
ENHANCEMENT_STD_MULTIPLIER = 1.5 # hotspot if enhancement > background_std * multiplier


def init_gee(project_id: str):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def get_ch4_at_point(lon: float, lat: float, date_str: str, window_days: int = CH4_DATE_WINDOW_DAYS):
    """
    Return (ch4_value, ch4_qa, pixel_id, pixel_date) for the closest-in-time,
    QA-passing S5P CH4 retrieval at (lon, lat). Returns Nones if no valid
    retrieval is found within the window.
    """
    point = ee.Geometry.Point([lon, lat])
    center_date = ee.Date(date_str)
    start = center_date.advance(-window_days, "day")
    end = center_date.advance(window_days, "day")

    coll = (
        ee.ImageCollection(S5P_COLLECTION)
        .filterBounds(point)
        .filterDate(start, end)
    )

    def add_time_diff(img):
        diff = ee.Number(img.get("system:time_start")).subtract(
            ee.Number(center_date.millis())
        ).abs()
        return img.set("time_diff", diff)

    coll = coll.map(add_time_diff).sort("time_diff")
    if coll.size().getInfo() == 0:
        return None, None, None, None

    img = coll.first()
    qa = img.select(QA_BAND).reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=7000, maxPixels=1e6
    ).get(QA_BAND).getInfo()

    if qa is None or qa < MIN_QA:
        return None, None, None, None

    ch4 = img.select(CH4_BAND).reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=7000, maxPixels=1e6
    ).get(CH4_BAND).getInfo()

    pixel_id = img.get("system:index").getInfo()
    pixel_date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    return ch4, qa, pixel_id, pixel_date


def get_local_background(center_lon: float, center_lat: float, date_str: str,
                          radius_m: int = BACKGROUND_RADIUS_M,
                          window_days: int = BACKGROUND_DATE_WINDOW_DAYS):
    """
    Estimate the local CH4 background as the median CH4 over a ring/disc
    surrounding the site (radius_m) and a wider time window, using
    QA-filtered retrievals. Returns (background_median, background_std).
    """
    center = ee.Geometry.Point([center_lon, center_lat])
    region = center.buffer(radius_m)
    d0 = ee.Date(date_str).advance(-window_days, "day")
    d1 = ee.Date(date_str).advance(window_days, "day")

    coll = (
        ee.ImageCollection(S5P_COLLECTION)
        .filterBounds(region)
        .filterDate(d0, d1)
        .map(lambda img: img.updateMask(img.select(QA_BAND).gte(MIN_QA)))
        .select(CH4_BAND)
    )

    stats = coll.reduce(ee.Reducer.median().combine(ee.Reducer.stdDev(), sharedInputs=True)) \
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=7000, maxPixels=1e7)

    median_val = stats.get(f"{CH4_BAND}_median").getInfo()
    std_val = stats.get(f"{CH4_BAND}_stdDev").getInfo()
    return median_val, std_val


def label_patches(extraction_log_path: str, gee_project: str, out_path: str,
                   site_lon: float, site_lat: float):
    init_gee(gee_project)

    log_df = pd.read_csv(extraction_log_path)
    completed = log_df[log_df["status"] == "COMPLETED"].copy()
    print(f"Labeling {len(completed)} genuinely completed patches "
          f"(dropping {len(log_df) - len(completed)} incomplete/rejected/no-scene rows).")

    # One background estimate per unique requested_date to avoid recomputing per-row.
    unique_dates = completed["requested_date"].unique()
    background_by_date = {}
    for d in unique_dates:
        median_val, std_val = get_local_background(site_lon, site_lat, d)
        background_by_date[d] = (median_val, std_val)
        print(f"  background for {d}: median={median_val}, std={std_val}")

    records = []
    for _, row in completed.iterrows():
        ch4, qa, pixel_id, pixel_date = get_ch4_at_point(
            row["longitude"], row["latitude"], row["requested_date"]
        )
        bg_median, bg_std = background_by_date.get(row["requested_date"], (None, None))

        enhancement = None
        label_enh = None
        if ch4 is not None and bg_median is not None:
            enhancement = ch4 - bg_median
            threshold = (bg_std or 0) * ENHANCEMENT_STD_MULTIPLIER
            label_enh = int(enhancement > threshold)

        records.append(dict(
            patch_id=row["patch_id"], latitude=row["latitude"], longitude=row["longitude"],
            patch_date=row["matched_scene_date"],
            ch4_value=ch4, ch4_qa=qa, ch4_pixel_id=pixel_id, ch4_pixel_date=pixel_date,
            ch4_background_median=bg_median, ch4_background_std=bg_std,
            ch4_enhancement=enhancement, label_enhancement=label_enh,
        ))

    meta = pd.DataFrame(records)

    # Reference-only global percentile label, for documented comparison against
    # the legacy method — NOT used as the primary training label.
    valid_ch4 = meta["ch4_value"].dropna()
    p75 = valid_ch4.quantile(0.75)
    meta["label_percentile75"] = (meta["ch4_value"] > p75).astype("Int64")

    n_no_ch4 = meta["ch4_value"].isna().sum()
    if n_no_ch4:
        print(f"Warning: {n_no_ch4} patches had no QA-passing CH4 retrieval within "
              f"{CH4_DATE_WINDOW_DAYS} days and will need to be dropped before training.")

    meta.to_csv(out_path, index=False)

    valid = meta.dropna(subset=["label_enhancement"])
    print(f"\nFinal labeled patches: {len(valid)}")
    print(valid["label_enhancement"].value_counts(normalize=True).rename("enhancement_label_share"))
    print(meta["label_percentile75"].value_counts(normalize=True).rename("percentile75_label_share (reference only)"))
    print(f"\nWritten to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Associate CH4 values and assign hotspot labels.")
    parser.add_argument("--extraction-log", required=True)
    parser.add_argument("--gee-project", required=True)
    parser.add_argument("--out", default="data/processed/metadata.csv")
    parser.add_argument("--site-lon", type=float, required=True,
                         help="Longitude of the Vellakkal dumpsite (background reference center)")
    parser.add_argument("--site-lat", type=float, required=True,
                         help="Latitude of the Vellakkal dumpsite (background reference center)")
    args = parser.parse_args()

    label_patches(args.extraction_log, args.gee_project, args.out, args.site_lon, args.site_lat)


if __name__ == "__main__":
    main()
