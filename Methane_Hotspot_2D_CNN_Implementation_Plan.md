# AI-Based Methane Hotspot Detection — Implementation Plan
### Vellakkal Dumpsite, Madurai, Tamil Nadu | Sentinel-5P + Sentinel-2 | 2D CNN

---

## 1. Current Dataset Analysis

What you have today is a **point-based tabular export**, not an imagery dataset:

- 6,200 rows, each row = one (lat, lon, date) location where GEE sampled pixel/band values.
- Columns are **scalar values at a single pixel**, not a spatial neighborhood: `B2, B3, B4, B8, B11, B12, NDVI` are single numbers per row, and `CH4, CH4_uncertainty` are single TROPOMI values per row.
- 43 distinct months are represented (Feb 2020 – Dec 2025), so there is real temporal spread, but spatial diversity is unknown until we inspect how the 6,200 points are distributed geographically (grid sample vs. random sample vs. repeated fixed points across time).
- CH4 mean ≈ 1919.31 ppb, std ≈ 24.27, range 1843–2019 ppb — a fairly tight, plausible TROPOMI CH4 column range for this region.
- Label was derived post-hoc from the 75th percentile (~1934.93 ppb) → 25% hotspot / 75% non-hotspot. This is a **relative, sample-dependent threshold**, not a physical/absolute methane enhancement threshold — this must be revisited (Section 5).
- The 1D CNN baseline trained on this tabular data performs *worse than a trivial majority-class classifier* on accuracy (41.66% vs. 75% if it always predicted "non-hotspot"), while precision is high and recall is very low. This pattern typically means the model is only rarely predicting the positive class, and when it does, it's often right — but it is missing most real hotspots. This is expected from 7 scalar features with no spatial context, and is not something to fix by tuning the 1D model further; the fix is to move to genuine spatial patches (2D CNN), which is already the plan.

**Conclusion:** The CSV is useful as a **label source and geolocation index**, but cannot itself become a 2D CNN dataset. Every genuine patch must be freshly pulled from Sentinel-2 imagery in GEE, centered on real coordinates, for a real acquisition date/window.

---

## 2. What Is Missing From the Current CSV for a True 2D CNN

| Missing element | Why it matters |
|---|---|
| Actual pixel neighborhoods (32×32 spatial windows) | CNNs need spatial context (edges, texture, surrounding land cover) — a single pixel vector cannot provide this |
| Confirmed one-to-one mapping between each row's `s2_images` / `s5p_images` ID and a specific GEE image | Needed to re-fetch the *exact* same image referenced in each row, not just "an image near that date" |
| Explicit geographic footprint / bounding box per sample | Needed to check for spatial overlap between samples (leakage risk) |
| Explicit acquisition date separation from "label date" | Sentinel-2 revisit is ~5 days, Sentinel-5P TROPOMI is daily but coarse (~7×5.5 km, later ~5.5×7 km); you need a defined temporal matching window (Section 4) |
| Cloud/QA mask information for Sentinel-2 | Needed to discard cloudy patches that would corrupt spectral values |
| A record of which points are spatially close to each other (nearest-neighbor distance) | Needed to prevent near-duplicate patches from leaking across train/val/test |

---

## 3. Obtaining Genuine Sentinel-2 Image Patches from GEE

**Approach:** For each of the 6,200 (lat, lon, date) rows (or a curated subset — see Section 6), re-query Earth Engine directly using the Python API (`earthengine-api`), not the CSV values.

Steps:
1. **Re-authenticate** the same GEE project used for the original export.
2. **Reconstruct each sample's Sentinel-2 image**: filter `COPERNICUS/S2_SR_HARMONIZED` by the sample's date (±5 day window), bounds (small buffer around lat/lon), and cloud cover (`CLOUDY_PIXEL_PERCENTAGE < 20`), then pick the least-cloudy scene, matching the same logic implied by the `s2_images` ID field where possible.
3. **Define a fixed pixel window**: 32×32 pixels at 10 m resolution (bands B2/B3/B4/B8) → a 320 m × 320 m footprint. Note B11/B12 are natively 20 m — see Section 9 for resampling.
4. **Clip/sample the neighborhood**: use `ee.Image.neighborhoodToArray()` or `ee.Image.sampleRectangle()` centered on the point, sized to 32×32.
5. **Attach a cloud/QA mask check**: reject patches where the QA60 or SCL band shows >10% cloud/shadow/snow within the 32×32 window.
6. **Export as GeoTIFF (not CSV)**: batch export via `ee.batch.Export.image.toDrive()`, one small GeoTIFF per patch (or tiled export + later cropping), to Google Drive, then sync/download into Colab.
7. **Track provenance**: log source image ID, acquisition date, and exact center coordinates for every exported patch, so every patch is traceable back to a genuine satellite acquisition.

This guarantees every patch is a real, freshly-pulled spatial image — not a reconstruction from scalar CSV values.

---

## 4. Associating Sentinel-5P CH4 Values with Each Patch

Sentinel-5P TROPOMI CH4 (`COPERNICUS/S5P/OFFL/L3_CH4`, band `CH4_column_volume_mixing_ratio_dry_air`) has ~7×5.5 km native resolution — far coarser than a 320 m Sentinel-2 patch. This means:

- **Many Sentinel-2 patches will share the same underlying CH4 pixel.** This is expected and must be documented, not hidden — it is a genuine physical constraint of combining these two sensors, not a data-quality flaw.
- Matching procedure per patch:
  1. Take the patch center coordinate.
  2. Extract the CH4 (and `CH4_uncertainty`) value from the S5P image whose acquisition date is closest to the Sentinel-2 acquisition date, within a defined tolerance window (recommend ±1–3 days, documented explicitly).
  3. If S5P has a QA band (`qa_value`), filter out low-quality retrievals (`qa_value < 0.5` is the standard TROPOMI CH4 recommendation).
  4. Record which S5P pixel (by its own coordinates/ID) supplied the CH4 value, so it's traceable which Sentinel-2 patches share a CH4 source — this becomes important for leakage-aware splitting (Section 8).

---

## 5. Correct Hotspot Labeling Strategy

The current 75th-percentile threshold is a **relative, dataset-specific cutoff**, not a scientifically grounded methane-enhancement threshold. Recommended refinement:

- **Local background subtraction instead of global percentile**: Compute a local/regional CH4 background (e.g., median CH4 over a broader time window and a larger surrounding area excluding the dumpsite), then define enhancement as `CH4_local − CH4_background`. A hotspot label is triggered when this enhancement exceeds a defined threshold (e.g., background + 1.5×std, or a literature-informed ppb enhancement value for landfill CH4 plumes).
- Keep the **percentile-based threshold as a documented fallback/sensitivity check**, not the primary method — report both so the labeling choice is transparent and defensible.
- **Uncertainty-aware filtering**: exclude/down-weight samples where `CH4_uncertainty` is unusually high, since a "hotspot" driven by noise is a false label.
- Document the final chosen threshold value, its physical justification, and how many patches fall into each class **after** this relabeling (expect the ratio to shift from the current 75/25 split).

---

## 6. Patch Size and Number of Patches

- **Patch size**: 32×32 pixels, 7 channels (B2, B3, B4, B8, B11, B12, NDVI) → input shape `32×32×7`. This is a good balance: large enough to capture surrounding land-cover context around the dumpsite, small enough to keep compute manageable and avoid mixing in unrelated land-use far from the point of interest.
- **Target count**: 8,000–10,000 **genuine, non-duplicated** patches. Given only 6,200 original rows, reaching this target honestly requires:
  - Using **all valid, cloud-free (lat, lon, date) rows** as patch centers (expect some loss from cloud/QA filtering — realistically maybe 4,500–5,800 usable).
  - **Legitimately expanding spatial coverage**: sampling additional genuine grid points around the dumpsite and its surrounding control/background area (not just the original 6,200 points), across the same 2020–2025 time range, using the same GEE pipeline. This adds real new observations rather than duplicating existing ones.
  - **Legitimately expanding temporal coverage**: for existing high-value locations (e.g., near the dumpsite boundary), pulling additional cloud-free Sentinel-2 acquisitions across different valid months, each paired with its own genuinely matched CH4 value.
  - If, after all genuine expansion, the count still falls short of 8,000–10,000, that's an acceptable and honestly documented outcome — **the final report must state the true number of genuine patches obtained**, and must not pad the number with duplicates. A well-documented 5,000–7,000 genuine patch dataset is scientifically stronger than a padded 10,000 one.

---

## 7. Dataset Balancing

- After relabeling (Section 5), assess the new class ratio.
- Preferred balancing approach (in order of preference):
  1. **Collect more genuine minority-class patches** by targeting known/likely hotspot zones and time windows for additional sampling (still real acquisitions, not synthetic).
  2. **Class weighting during training** (`class_weight` in Keras) instead of resampling — preserves all genuine data.
  3. **Mild undersampling of the majority class** if needed for a specific experiment — but keep and document the untouched full dataset as well.
  4. **Do NOT use image-level SMOTE/synthetic oversampling on raw pixel data** — this fabricates spectral values and violates the "do not fabricate satellite data" requirement. Data augmentation (Section 9) is different and acceptable — it applies label-preserving geometric transforms to genuine patches, not synthetic content generation.

---

## 8. Train/Validation/Test Splitting Without Leakage

Random row-level splitting is invalid here because neighboring patches (same location, nearby dates) are highly spatially/temporally correlated, and many patches share the same underlying CH4 pixel.

Recommended strategy — **spatial block splitting**:
1. Divide the study area into geographic blocks/tiles (e.g., a grid over the dumpsite + surrounding region, block size larger than the CH4 footprint, ~7×5.5 km or bigger).
2. Assign each **entire block** to only one of train/val/test — never split a block's patches across sets.
3. Aim for roughly 70/15/15 block-level allocation, checked afterward at the patch level to confirm the resulting sample split is reasonably balanced.
4. **Secondary temporal check**: within any block, also avoid placing the same exact location's different-month patches across different sets when possible (or explicitly document if this cannot be fully avoided due to limited locations).
5. Record and report the number of unique locations and unique CH4 source-pixels in each split, to demonstrate leakage was controlled for, not just assumed.

---

## 9. Preprocessing and Normalization

- **Cloud/QA masking**: discard or flag patches with significant cloud/shadow/snow contamination (Section 3).
- **Band resampling**: B11/B12 (20 m) resampled to 10 m (bilinear) to align with B2/B3/B4/B8/NDVI at 32×32.
- **NDVI**: computed as `(B8 − B4) / (B8 + B4)`, guarded against divide-by-zero.
- **Reflectance scaling**: Sentinel-2 SR bands scaled from raw DN (0–10000) to reflectance (0–1) by dividing by 10000.
- **Per-channel normalization**: standardize each of the 7 channels (zero mean, unit variance) using statistics computed **only from the training split** (never from val/test, to avoid leakage), then applied to val/test.
- **CH4/CH4_uncertainty** (used for labeling, not as CNN input): documented separately, not fed into the CNN itself, to keep Sentinel-5P strictly as the label source and Sentinel-2 strictly as the model input — this keeps the "coarse vs. high-resolution" data roles clearly distinguished, as required.
- **Data augmentation** (train set only): random flips/rotations (90°/180°/270°), since methane hotspot patterns have no fixed orientation — this is label-preserving and does not fabricate new spectral content.

---

## 10. 2D CNN Architecture

Start with a **custom, compact 2D CNN** appropriate for 32×32×7 inputs and a dataset in the thousands (not millions) of samples — avoiding a network so deep it overfits:

- Input: `32×32×7`
- Block 1: Conv2D(32, 3×3) → BatchNorm → ReLU → Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout
- Block 2: Conv2D(64, 3×3) → BatchNorm → ReLU → Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout
- Block 3: Conv2D(128, 3×3) → BatchNorm → ReLU → GlobalAveragePooling2D
- Dense(64) → ReLU → Dropout
- Dense(1) → Sigmoid (binary: hotspot / non-hotspot)

L2 regularization on conv/dense layers, dropout 0.3–0.5, batch norm throughout to stabilize training on a modest dataset size.

**Transfer learning comparison (optional, secondary experiment)**: ResNet18/EfficientNet-B0 adapted to accept 7-channel input (replace first conv layer, since these architectures are pretrained on 3-channel RGB) — only worthwhile as a comparison once the custom CNN baseline is established, and must be clearly framed as an exploratory comparison, not the primary model, given the channel mismatch with pretrained weights.

---

## 11. Training Strategy

- Loss: binary cross-entropy (with `class_weight` if not otherwise balanced).
- Optimizer: Adam, initial LR ~1e-3 with `ReduceLROnPlateau` or cosine decay.
- Callbacks: `EarlyStopping` (monitor val loss/AUC, patience ~10–15), `ModelCheckpoint` (save best val AUC), `ReduceLROnPlateau`.
- Batch size: 32–64 (small dataset, small patches).
- Epochs: up to 100–150 with early stopping controlling actual length.
- K-fold or repeated-holdout **cross-validation at the block level** (not row level) recommended as a robustness check, given the modest dataset size.
- Track training/validation curves (loss, accuracy, AUC) to detect overfitting early, given the limited sample count.

---

## 12. Evaluation Metrics

On the held-out spatial test split:
- Confusion matrix
- Accuracy, Precision, Recall, F1-score (per class, plus macro-average given class imbalance)
- ROC-AUC and ROC curve
- Precision-Recall curve and PR-AUC (more informative than ROC-AUC under class imbalance)
- Calibration check (predicted probability vs. observed frequency) — useful before treating output as a "hotspot probability map"
- Explicit comparison table against the 1D CNN baseline, framed honestly (different data modality, not directly apples-to-apples, but useful context)

---

## 13. Model Saving

- Save the trained Keras model in `.keras` (native) format, plus a `.h5` fallback for compatibility.
- Save the exact preprocessing parameters used (per-channel mean/std, band order, patch size) in a companion `preprocessing_config.json`, since inference must replicate preprocessing exactly.
- Save the label-threshold definition (Section 5) alongside the model, since the model's target definition is part of its identity.
- Version the model directory with a date/run-ID (e.g., `models/cnn_v1_2026-08-09/`).

---

## 14. Inference on New Satellite Data

- Pipeline: new AOI/date → re-run the same GEE patch-extraction logic (Section 3) → same preprocessing/normalization (using saved training statistics, not re-computed) → model prediction → probability per patch.
- Batch inference over a grid of new points around the dumpsite (or any new AOI) to produce a spatially continuous prediction surface.
- Output: per-patch center coordinate + predicted probability + predicted class, saved as a table (CSV) and as a raster (GeoTIFF) for GIS use.
- Clearly log the acquisition date(s) used for inference, since predictions are tied to a specific observation window, not a permanent site classification.

---

## 15. GIS Visualization

- Convert patch-center predictions into a point layer (CSV/GeoJSON: lat, lon, probability, class) importable into QGIS.
- Optionally interpolate/rasterize point predictions into a continuous probability heatmap (e.g., IDW or simple grid rasterization) for visual clarity — clearly labeled as an interpolation, not raw sensor data.
- Overlay on Sentinel-2 true-color basemap and/or OpenStreetMap for context around Vellakkal dumpsite.
- Optional: GEE-side visualization (Code Editor or `geemap` in Python/Colab) for a quick in-notebook interactive map, in addition to the QGIS deliverable.

---

## 16. Final Project Folder Structure

```
methane-hotspot-detection/
│
├── data/
│   ├── raw/
│   │   ├── original_gee_export.csv
│   │   └── patches_geotiff/            # one GeoTIFF per genuine patch
│   ├── processed/
│   │   ├── patches_array.npz           # stacked 32x32x7 arrays + labels + metadata
│   │   └── metadata.csv                # patch_id, lat, lon, date, CH4, CH4_unc, block_id, split
│   └── splits/
│       ├── train_ids.txt
│       ├── val_ids.txt
│       └── test_ids.txt
│
├── notebooks/
│   ├── 01_gee_patch_extraction.ipynb
│   ├── 02_ch4_association_labeling.ipynb
│   ├── 03_eda_and_balance_check.ipynb
│   ├── 04_train_cnn.ipynb
│   └── 05_inference_and_gis_export.ipynb
│
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
│
├── models/
│   └── cnn_v1_<date>/
│       ├── model.keras
│       ├── preprocessing_config.json
│       └── label_threshold.json
│
├── outputs/
│   ├── figures/                        # confusion matrix, ROC, PR curves, training curves
│   ├── predictions/                    # inference CSV/GeoTIFF outputs
│   └── gis/                            # QGIS-ready layers
│
├── reports/
│   └── final_report.md / .docx
│
├── requirements.txt
└── README.md
```

---

## 17. Required Python Packages

```
earthengine-api
geemap
numpy
pandas
rasterio
geopandas
shapely
scikit-learn
tensorflow          # includes keras
matplotlib
seaborn
tqdm
pillow
```
Plus Google Colab's built-in `google.colab` (Drive mounting) and `google-auth` for GEE authentication.

---

## 18. How GEE, Google Drive, and Colab Interact

1. **GEE (Python API in Colab)**: authenticate once per session (`ee.Authenticate()` + `ee.Initialize(project=...)`); this is where all real satellite data queries and exports originate.
2. **Export step**: GEE batch export tasks (`Export.image.toDrive`) write GeoTIFF patches (or larger tiles to be cropped later) into a designated Google Drive folder — exports run server-side on Google's infrastructure and must be polled for completion (`ee.batch.Task.status()`), not downloaded instantly.
3. **Google Drive**: acts as the intermediate storage layer between GEE's export jobs and Colab's local runtime; the same Google account should own both the GEE project and the Drive folder.
4. **Colab**: mounts the same Google Drive (`drive.mount('/content/drive')`), reads the exported GeoTIFFs from there, performs stacking into `.npz`/array form, runs preprocessing, training, evaluation, and writes all outputs (models, figures, prediction layers) back into Drive for persistence beyond the Colab session.
5. **Loop-back for inference**: new AOI/date requests re-enter at step 1 (GEE query) → step 2 (export) → step 4 (Colab inference), keeping the same pipeline for both training-data creation and future inference.

---

## File-by-File Breakdown

| File | Purpose | Input | Output | Dependencies | Approx. size |
|---|---|---|---|---|---|
| `src/gee_extraction.py` | Query GEE, filter cloud/QA, export 32×32×7 patches as GeoTIFFs with provenance log | `original_gee_export.csv` (coords/dates), GEE assets | `patches_geotiff/*.tif`, `extraction_log.csv` | `earthengine-api`, `geemap`, `pandas` | ~150–200 lines |
| `src/ch4_labeling.py` | Match each patch to nearest-in-time, QA-filtered S5P CH4 value; compute background-subtracted enhancement; assign labels | `extraction_log.csv`, GEE S5P collection | `metadata.csv` with CH4, enhancement, label | `earthengine-api`, `pandas`, `numpy` | ~100–150 lines |
| `src/preprocessing.py` | Load GeoTIFFs, resample B11/B12, compute NDVI, scale reflectance, stack into arrays | `patches_geotiff/*.tif` | `patches_array.npz` | `rasterio`, `numpy` | ~120–170 lines |
| `src/dataset_split.py` | Build geographic blocks, assign blocks to train/val/test, check leakage | `metadata.csv` | `train_ids.txt`, `val_ids.txt`, `test_ids.txt` | `geopandas`, `shapely`, `scikit-learn` | ~100–140 lines |
| `src/model.py` | Define the 2D CNN architecture (and optional transfer-learning variant) | — | Keras model object | `tensorflow` | ~80–120 lines |
| `src/train.py` | Load split data, normalize (train-stats only), train model, apply callbacks | `patches_array.npz`, split files | `models/cnn_v1_<date>/model.keras`, training curves | `tensorflow`, `numpy`, `matplotlib` | ~150–200 lines |
| `src/evaluate.py` | Run test-set inference, compute all metrics, generate plots | trained model, test split | `outputs/figures/*.png`, metrics report | `scikit-learn`, `matplotlib`, `seaborn` | ~120–160 lines |
| `src/inference.py` | Run pipeline on new AOI/date for deployment-style prediction | new coordinates/date, saved model + config | `outputs/predictions/*.csv/.tif` | `tensorflow`, `rasterio`, `earthengine-api` | ~100–150 lines |
| `src/gis_export.py` | Convert predictions to GeoJSON/rasterized layers for QGIS | predictions CSV | `outputs/gis/*.geojson`, `.tif` | `geopandas`, `rasterio` | ~80–110 lines |
| `notebooks/01_gee_patch_extraction.ipynb` | Interactive driver for extraction step, sanity checks/visual QA of patches | — | Calls `gee_extraction.py` | `geemap` for map preview | ~short, mostly calls |
| `notebooks/02_ch4_association_labeling.ipynb` | Interactive driver for CH4 matching + label distribution review | — | Calls `ch4_labeling.py` | — | short |
| `notebooks/03_eda_and_balance_check.ipynb` | Explore class balance, spatial distribution, sample patch visualization | `metadata.csv`, `patches_array.npz` | figures | `matplotlib`, `seaborn` | short–medium |
| `notebooks/04_train_cnn.ipynb` | Interactive driver for training + evaluation | — | Calls `train.py`, `evaluate.py` | — | short |
| `notebooks/05_inference_and_gis_export.ipynb` | Interactive driver for inference + GIS export on new data | — | Calls `inference.py`, `gis_export.py` | `geemap` for preview | short |
| `README.md` | Project overview, setup, how to reproduce each stage | — | — | — | ~1–2 pages |
| `requirements.txt` | Pinned package list | — | — | — | ~15 lines |

---

## Batch Plan for Progressive File Generation

To stay within token-safe limits, we'll generate the project in the following batches, each a separate response, building in dependency order:

- **Batch 1** — `src/gee_extraction.py` (patch extraction) + `notebooks/01_gee_patch_extraction.ipynb`
- **Batch 2** — `src/ch4_labeling.py` (CH4 matching + labeling) + `notebooks/02_ch4_association_labeling.ipynb`
- **Batch 3** — `src/preprocessing.py` + `notebooks/03_eda_and_balance_check.ipynb`
- **Batch 4** — `src/dataset_split.py` (leakage-safe splitting)
- **Batch 5** — `src/model.py` + `src/train.py` + `notebooks/04_train_cnn.ipynb`
- **Batch 6** — `src/evaluate.py` (metrics, confusion matrix, ROC/PR curves)
- **Batch 7** — `src/inference.py` + `src/gis_export.py` + `notebooks/05_inference_and_gis_export.ipynb`
- **Batch 8** — `README.md`, `requirements.txt`, final folder assembly

Each batch will be delivered only when you confirm readiness to proceed, so you can review/adjust before the next stage builds on it.
