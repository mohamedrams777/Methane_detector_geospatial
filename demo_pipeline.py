"""
demo_pipeline.py

Master End-to-End Execution Pipeline for:
AI-Based Landfill Methane Hotspot Detection, Downscaling, 7-Day Forecasting, and 3D Visualization.

Executes:
1. Dataset Preparation & Spatial Block Splitting (data/processed, data/splits)
2. Multi-Spectral ResNet-50 Training with Squeeze-and-Excitation Attention
3. 7-Day Spatio-Temporal Methane Hotspot Forecaster Training
4. Test-set Evaluation (ROC-AUC, PR-AUC, Confusion Matrix, Calibration)
5. QGIS-Ready GIS Layer Generation (GeoJSON point layer + Interpolated Raster)
6. Standalone Interactive 3D WebGL Elevation & Plume Map Generation
"""

import os
import sys
import subprocess
import time


def print_step(title):
    print("\n" + "=" * 70)
    print(f"  >>> {title}")
    print("=" * 70 + "\n")


def run_pipeline():
    start_time = time.time()

    # Step 1: Data Preparation
    print_step("Step 1/6: Generating / Checking Dataset & Leakage-Safe Splits")
    from generate_sample_data import generate_synthetic_multispectral_dataset
    generate_synthetic_multispectral_dataset(num_samples=1000, patch_size=32)

    # Step 2: Multi-Spectral ResNet-50 Training
    print_step("Step 2/6: Training Multi-Spectral ResNet-50 (MS-ResNet50)")
    from train_resnet import train_ms_resnet
    model_dir = train_ms_resnet(
        patches_path="data/processed/patches_array.npz",
        splits_dir="data/splits",
        models_dir="models",
        epochs=4,
        batch_size=32,
        learning_rate=5e-4,
    )

    # Step 3: 7-Day Spatio-Temporal Forecasting Model
    print_step("Step 3/6: Training 7-Day Spatio-Temporal Hotspot Forecaster")
    from forecasting import train_spatiotemporal_forecaster
    train_spatiotemporal_forecaster(epochs=5, batch_size=16)

    # Step 4: Full Test-Set Evaluation & Baseline Comparison
    print_step("Step 4/6: Evaluating Model & Generating Metrics Curves")
    from evaluate import run_evaluation
    # Run evaluation script on the newly trained MS-ResNet50 run
    subprocess.run([
        sys.executable, "evaluate.py",
        "--patches", "data/processed/patches_array.npz",
        "--splits-dir", "data/splits",
        "--model-dir", model_dir,
    ], check=True)

    # Step 5: GIS Layer Generation
    print_step("Step 5/6: Exporting QGIS-Ready Layers (GeoJSON & GeoTIFF)")
    from gis_export import join_with_metadata, export_points_geojson, idw_interpolate
    preds_csv = "outputs/predictions/test_predictions.csv"
    meta_csv = "data/processed/metadata_with_splits.csv"
    if os.path.exists(preds_csv) and os.path.exists(meta_csv):
        df_merged = join_with_metadata(preds_csv, meta_csv)
        os.makedirs("outputs/gis", exist_ok=True)
        export_points_geojson(df_merged, "outputs/gis/predictions_points.geojson")
        idw_interpolate(df_merged, "outputs/gis/predictions_probability_surface.tif")

    # Step 6: Interactive 3D WebGL Elevation & Methane Map
    print_step("Step 6/6: Generating Interactive 3D Elevation & Plume Map")
    from visualize_3d_map import generate_3d_interactive_map
    map_path = "outputs/gis/vellakkal_3d_methane_map.html"
    generate_3d_interactive_map(output_html_path=map_path)

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.1f}s!")
    print("#" * 70)
    print("\nGenerated Artifacts & Outputs:")
    print(f"  1. Multi-Spectral ResNet-50 Model : {model_dir}/model.keras")
    print(f"  2. 7-Day Forecaster Model         : models/forecaster/forecaster.keras")
    print(f"  3. Training Curves                : outputs/figures/ms_resnet50_training_curves.png")
    print(f"  4. Confusion Matrix & ROC/PR      : outputs/figures/roc_curve.png, confusion_matrix.png")
    print(f"  5. 7-Day Forecast Progression     : outputs/figures/7day_forecast_progression.png")
    print(f"  6. QGIS GeoJSON & GeoTIFF Raster  : outputs/gis/predictions_points.geojson")
    print(f"  7. Interactive 3D WebGL Map       : {map_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
