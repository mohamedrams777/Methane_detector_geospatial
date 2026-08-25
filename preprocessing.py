"""
preprocessing.py

BATCH 3 — Load exported GeoTIFF patches, verify/align bands, compute final
per-patch arrays, and stack them into a single .npz dataset paired with the
labels from metadata.csv (Batch 2).

Purpose
-------
Turn the individual 32x32x7 GeoTIFF patches (Batch 1) plus their labels
(Batch 2) into one array file that train.py can load directly. This is also
where reflectance scaling and a final QA re-check happen (belt-and-braces on
top of the cloud filtering already done during extraction).

Input
-----
- Google Drive folder of exported GeoTIFFs (one per patch_id), e.g.
  data/raw/patches_geotiff/<patch_id>.tif
- data/processed/metadata.csv (from ch4_labeling.py)

Output
------
- data/processed/patches_array.npz containing:
    X          : float32 array, shape (N, 32, 32, 7)  [B2,B3,B4,B8,B11,B12,NDVI]
    y          : int array, shape (N,)                 [label_enhancement]
    patch_ids  : str array, shape (N,)
- data/processed/metadata_final.csv : metadata.csv rows that made it into X/y,
  in the same order as X/y, with any additional QA-drop reasons appended.

Dependencies
------------
rasterio, numpy, pandas, tqdm

Usage
-----
    python preprocessing.py \
        --geotiff-dir data/raw/patches_geotiff \
        --metadata data/processed/metadata.csv \
        --out data/processed/patches_array.npz \
        --patch-size 32
"""

import argparse
import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI"]
REFLECTANCE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # NDVI already in [-1, 1]
DN_SCALE = 10000.0
MAX_MISSING_PIXEL_FRACTION = 0.01  # reject patch if >1% of pixels are nodata/NaN


def load_patch(tif_path: str, patch_size: int):
    """
    Read a GeoTIFF patch, verify band count/order and dimensions, apply
    reflectance scaling, and return a (patch_size, patch_size, 7) float32
    array, or None if the patch fails validation.
    """
    with rasterio.open(tif_path) as src:
        if src.count != len(BAND_ORDER):
            return None, f"unexpected_band_count_{src.count}"

        arr = src.read().astype("float32")  # shape (bands, H, W)
        arr = np.transpose(arr, (1, 2, 0))   # -> (H, W, bands)

        h, w, _ = arr.shape
        if h != patch_size or w != patch_size:
            # Center-crop or pad if the export came back a pixel or two off
            # due to reprojection rounding; reject if the mismatch is large.
            if abs(h - patch_size) > 2 or abs(w - patch_size) > 2:
                return None, f"bad_dimensions_{h}x{w}"
            arr = _center_fit(arr, patch_size)

        missing_frac = np.isnan(arr).mean()
        if missing_frac > MAX_MISSING_PIXEL_FRACTION:
            return None, f"too_many_missing_pixels_{missing_frac:.3f}"
        arr = np.nan_to_num(arr, nan=0.0)

        # Reflectance scaling for the 6 spectral bands (indices 0-5); NDVI
        # (index 6) is left as-is since it's already a bounded ratio.
        arr[..., :6] = arr[..., :6] / DN_SCALE

        return arr, None


def _center_fit(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Center-crop or zero-pad a slightly mis-sized patch to exactly patch_size x patch_size."""
    h, w, c = arr.shape
    out = np.zeros((patch_size, patch_size, c), dtype=arr.dtype)
    ch, cw = min(h, patch_size), min(w, patch_size)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    ty0, tx0 = (patch_size - ch) // 2, (patch_size - cw) // 2
    out[ty0:ty0 + ch, tx0:tx0 + cw, :] = arr[y0:y0 + ch, x0:x0 + cw, :]
    return out


def build_dataset(geotiff_dir: str, metadata_path: str, out_path: str, patch_size: int):
    meta = pd.read_csv(metadata_path)
    meta = meta.dropna(subset=["ch4_value", "label_enhancement"]).reset_index(drop=True)
    print(f"Metadata rows with a valid CH4 value and label: {len(meta)}")

    X_list, y_list, id_list, drop_reasons = [], [], [], []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Loading patches"):
        patch_id = row["patch_id"]
        tif_path = os.path.join(geotiff_dir, f"{patch_id}.tif")

        if not os.path.exists(tif_path):
            drop_reasons.append((patch_id, "geotiff_not_found"))
            continue

        arr, reason = load_patch(tif_path, patch_size)
        if arr is None:
            drop_reasons.append((patch_id, reason))
            continue

        X_list.append(arr)
        y_list.append(int(row["label_enhancement"]))
        id_list.append(patch_id)

    X = np.stack(X_list, axis=0).astype("float32")
    y = np.array(y_list, dtype="int64")
    ids = np.array(id_list, dtype="object")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, patch_ids=ids)

    final_meta = meta[meta["patch_id"].isin(ids)].copy()
    final_meta["array_index"] = final_meta["patch_id"].apply(lambda pid: list(ids).index(pid))
    final_meta = final_meta.sort_values("array_index").reset_index(drop=True)
    final_meta.to_csv(out_path.replace("patches_array.npz", "metadata_final.csv"), index=False)

    print(f"\nFinal genuine dataset: {len(ids)} patches, shape {X.shape}")
    print(f"Label distribution: {pd.Series(y).value_counts(normalize=True).to_dict()}")
    if drop_reasons:
        drop_df = pd.DataFrame(drop_reasons, columns=["patch_id", "reason"])
        print(f"\nDropped {len(drop_df)} patches during preprocessing:")
        print(drop_df["reason"].value_counts())
        drop_log_path = out_path.replace("patches_array.npz", "preprocessing_drop_log.csv")
        drop_df.to_csv(drop_log_path, index=False)
        print(f"Drop log written to {drop_log_path}")

    print(f"\nDataset written to {out_path}")
    print("This is the true, honest count of genuine, non-duplicated patches "
          "available for splitting and training — report this number directly, "
          "do not pad it.")


def main():
    parser = argparse.ArgumentParser(description="Build the stacked patch array dataset.")
    parser.add_argument("--geotiff-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="data/processed/patches_array.npz")
    parser.add_argument("--patch-size", type=int, default=32)
    args = parser.parse_args()

    build_dataset(args.geotiff_dir, args.metadata, args.out, args.patch_size)


if __name__ == "__main__":
    main()
