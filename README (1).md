# AI-Based Methane Hotspot Detection
### Vellakkal Dumpsite, Madurai, Tamil Nadu — Sentinel-5P + Sentinel-2 — 2D CNN

## Overview

This project detects and classifies potential methane emission hotspots
around the Vellakkal dumpsite by combining:

- **Sentinel-5P TROPOMI** atmospheric CH4 column data (coarse, ~7×5.5 km) — used
  to derive hotspot labels
- **Sentinel-2** multispectral imagery (B2, B3, B4, B8, B11, B12 + NDVI, 10 m) —
  used as genuine 32×32×7 spatial patches, the 2D CNN's input

A 1D CNN baseline on tabular per-pixel CSV data was tested first (accuracy
≈41.7%, precision ≈80.5%, recall ≈17.4%, F1 ≈28.7%, ROC-AUC ≈59.6%) and
confirmed to be insufficient — it lacks spatial context entirely. The final
model is a custom 2D CNN trained on genuine Sentinel-2 image patches.

**No satellite data is fabricated or duplicated anywhere in this pipeline.**
Every patch is a fresh Earth Engine query traceable to a real Sentinel-2
scene ID and acquisition date. See `data/processed/preprocessing_drop_log.csv`
and `data/splits/leakage_report.txt` for full transparency on what was
dropped and how leakage was controlled for.

## Pipeline Stages

| Stage | Script | Notebook | What it does |
|---|---|---|---|
| 1 | `src/gee_extraction.py` | `01_gee_patch_extraction.ipynb` | Query GEE for genuine 32×32×7 Sentinel-2 patches, filter cloud/QA, export GeoTIFFs |
| 2 | `src/ch4_labeling.py` | `02_ch4_association_labeling.ipynb` | Match each patch to a QA-passing CH4 retrieval, compute background-subtracted enhancement, assign labels |
| 3 | `src/preprocessing.py` | `03_eda_and_balance_check.ipynb` | Stack patches into an array dataset, scale reflectance, run EDA/balance checks |
| 4 | `src/dataset_split.py` | — | Leakage-safe spatial block train/val/test split (70/15/15) |
| 5 | `src/model.py`, `src/train.py` | `04_train_cnn.ipynb` | Define and train the 2D CNN with train-only normalization and class weighting |
| 6 | `src/evaluate.py` | — | Confusion matrix, ROC/PR curves, calibration check, honest baseline comparison |
| 7 | `src/inference.py`, `src/gis_export.py` | `05_inference_and_gis_export.ipynb` | Score new points and export QGIS-ready layers |

Run stages in order — each depends on the previous stage's output.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need:
- A Google Earth Engine account with a registered Cloud project (pass its ID
  as `--gee-project` / `GEE_PROJECT` throughout)
- A Google Drive account (the same one used for GEE) with enough free space
  for exported GeoTIFF patches
- Google Colab (recommended) for running the notebooks, since they mount
  Drive directly — see `data_flow` below

## Data Flow: GEE ↔ Drive ↔ Colab

1. **GEE (Python API, from Colab)** — authenticate, query Sentinel-2/Sentinel-5P,
   submit batch export tasks.
2. **Export** — GEE runs exports server-side; poll `ee.batch.Task.status()`
   (handled by `check_export_status()` in `gee_extraction.py`) until complete.
3. **Google Drive** — the intermediate storage layer; exported GeoTIFFs land
   in a Drive folder shared between GEE and Colab.
4. **Colab** — mounts the same Drive, reads GeoTIFFs, runs preprocessing,
   training, evaluation, writes all outputs back to Drive.
5. **Inference loop-back** — new AOI/date requests re-enter at step 1.

## Folder Structure

```
methane-hotspot-detection/
├── data/
│   ├── raw/
│   │   ├── original_gee_export.csv       # source CSV (6,200 tabular rows)
│   │   ├── extraction_log.csv            # Batch 1 provenance log
│   │   └── patches_geotiff/              # genuine exported patches
│   ├── processed/
│   │   ├── metadata.csv                  # Batch 2 output (CH4 + labels)
│   │   ├── patches_array.npz             # Batch 3 output (X, y, patch_ids)
│   │   ├── metadata_final.csv            # Batch 3, aligned to array order
│   │   └── metadata_with_splits.csv      # Batch 4 output
│   └── splits/
│       ├── train_ids.txt / val_ids.txt / test_ids.txt
│       └── leakage_report.txt
├── notebooks/
│   ├── 01_gee_patch_extraction.ipynb
│   ├── 02_ch4_association_labeling.ipynb
│   ├── 03_eda_and_balance_check.ipynb
│   ├── 04_train_cnn.ipynb
│   └── 05_inference_and_gis_export.ipynb
├── src/
│   ├── gee_extraction.py
│   ├── ch4_labeling.py
│   ├── preprocessing.py
│   ├── dataset_split.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── gis_export.py
├── models/
│   └── cnn_v1_<date>/
│       ├── model.keras
│       ├── preprocessing_config.json
│       ├── label_threshold.json
│       └── training_history.csv
├── outputs/
│   ├── figures/       # training curves, confusion matrix, ROC/PR/calibration
│   ├── predictions/   # test + inference prediction CSVs
│   └── gis/           # QGIS-ready GeoJSON + interpolated raster
├── reports/
│   └── evaluation_metrics.json
├── requirements.txt
└── README.md
```

## Key Design Choices (and why)

- **Labeling**: primary label uses local background-subtracted CH4
  enhancement (25 km ring, ±30 day window), not a bare global percentile —
  the old percentile method is retained only as a documented reference
  column, not the training target.
- **Splitting**: geographic block splitting with union-find merging of any
  blocks sharing a CH4 source pixel — guarantees no shared TROPOMI retrieval
  or nearby location crosses a train/val/test boundary.
- **Normalization**: per-channel mean/std computed only on the training
  split, saved to `preprocessing_config.json`, and reused identically for
  val/test and later inference.
- **Balancing**: class weighting during training, not synthetic oversampling
  or row duplication — every sample in the dataset is a genuine observation.
- **Evaluation**: ROC-AUC, PR-AUC, and a calibration curve are all reported,
  since PR-AUC and calibration matter more than raw accuracy under class
  imbalance, especially before treating outputs as real probabilities.

## Honest Reporting Note

The number of genuine, non-duplicated patches that make it through the full
pipeline (cloud/QA filtering in Batch 1 and 3, valid CH4 match in Batch 2)
may land below the 8,000–10,000 target, depending on how much the original
6,200-row source and its ~254 unique locations / 12 unique Sentinel-2 scenes
can be genuinely expanded with new grid points and dates. **Report the true
final count from `metadata_final.csv`** — do not pad or duplicate to hit a
target number.
