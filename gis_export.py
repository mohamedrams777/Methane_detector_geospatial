"""
gis_export.py

BATCH 7 — Convert model predictions (test-set or new inference) into
GIS-ready layers: a point GeoJSON and an interpolated probability raster.

Purpose
-------
Produces QGIS-importable outputs from predictions produced by evaluate.py or
inference.py. The interpolated raster is clearly an interpolation of sparse
point predictions, not a raw sensor product, and is labeled as such.

Input
-----
- A predictions CSV with at least: latitude, longitude, hotspot_probability
  (evaluate.py's test_predictions.csv needs patch coordinates joined in first
  — see join_with_metadata(); inference.py's output already has coordinates.)

Output
------
- outputs/gis/predictions_points.geojson
- outputs/gis/predictions_probability_surface.tif  (IDW-interpolated raster)

Dependencies
------------
geopandas, shapely, rasterio, numpy, pandas, scipy
"""

import argparse
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree


def join_with_metadata(predictions_csv: str, metadata_csv: str) -> pd.DataFrame:
    """
    evaluate.py's test_predictions.csv has patch_id/y_prob but not lat/lon;
    join against metadata_with_splits.csv (from dataset_split.py) to recover
    coordinates before GIS export.
    """
    preds = pd.read_csv(predictions_csv)
    meta = pd.read_csv(metadata_csv)
    merged = preds.merge(meta[["patch_id", "latitude", "longitude"]], on="patch_id", how="left")
    merged = merged.rename(columns={"y_prob": "hotspot_probability", "y_pred": "hotspot_prediction"})
    return merged


def export_points_geojson(df: pd.DataFrame, out_path: str):
    df = df.dropna(subset=["latitude", "longitude", "hotspot_probability"])
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Point layer written to {out_path} ({len(gdf)} points)")


def idw_interpolate(df: pd.DataFrame, out_path: str, grid_res_deg: float = 0.002,
                     power: float = 2.0, k_neighbors: int = 8):
    """
    Inverse-distance-weighted interpolation of point hotspot_probability
    values onto a regular grid, written as a GeoTIFF. This is a smoothed
    visualization aid, not a raw measurement — labeled accordingly wherever
    it is displayed.
    """
    df = df.dropna(subset=["latitude", "longitude", "hotspot_probability"])
    lats, lons, vals = df["latitude"].values, df["longitude"].values, df["hotspot_probability"].values

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    # Pad the bounds slightly so edge points aren't right at the raster border.
    pad = grid_res_deg * 5
    lat_min, lat_max = lat_min - pad, lat_max + pad
    lon_min, lon_max = lon_min - pad, lon_max + pad

    grid_lats = np.arange(lat_max, lat_min, -grid_res_deg)
    grid_lons = np.arange(lon_min, lon_max, grid_res_deg)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lons, grid_lats)

    tree = cKDTree(np.column_stack([lons, lats]))
    query_points = np.column_stack([grid_lon_mesh.ravel(), grid_lat_mesh.ravel()])
    dists, idxs = tree.query(query_points, k=min(k_neighbors, len(vals)))

    dists = np.maximum(dists, 1e-10)
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    weights = 1.0 / (dists ** power)
    weighted_vals = np.sum(weights * vals[idxs], axis=1) / np.sum(weights, axis=1)
    surface = weighted_vals.reshape(grid_lat_mesh.shape).astype("float32")

    transform = from_origin(lon_min, lat_max, grid_res_deg, grid_res_deg)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=surface.shape[0], width=surface.shape[1],
        count=1, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(surface, 1)

    print(f"Interpolated probability surface written to {out_path} "
          f"(IDW, k={k_neighbors}, power={power}) — an interpolation for "
          f"visualization, not a raw sensor product.")


def main():
    parser = argparse.ArgumentParser(description="Export GIS-ready layers from model predictions.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--metadata-csv", default=None,
                         help="Required if predictions-csv lacks lat/lon (e.g. evaluate.py output)")
    parser.add_argument("--out-dir", default="outputs/gis")
    parser.add_argument("--grid-res-deg", type=float, default=0.002)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        if args.metadata_csv is None:
            raise ValueError("predictions-csv has no coordinates; pass --metadata-csv to join them in.")
        df = join_with_metadata(args.predictions_csv, args.metadata_csv)
    elif "hotspot_probability" not in df.columns and "y_prob" in df.columns:
        df = df.rename(columns={"y_prob": "hotspot_probability", "y_pred": "hotspot_prediction"})

    export_points_geojson(df, os.path.join(args.out_dir, "predictions_points.geojson"))
    idw_interpolate(df, os.path.join(args.out_dir, "predictions_probability_surface.tif"),
                     grid_res_deg=args.grid_res_deg)


if __name__ == "__main__":
    main()
