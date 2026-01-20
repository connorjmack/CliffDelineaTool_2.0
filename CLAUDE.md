# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CliffDelineaTool 2.0 is a deep learning system for detecting coastal cliff base and top positions from elevation transects. It builds on the original rule-based CliffDelineaTool by Swirad & Young (2022).

- **v1/** - Original MATLAB/Python implementation (rule-based, requires manual tuning)
- **v2/** - Deep learning implementation (CNN-BiLSTM, no manual tuning required)
- **datasets/** - Calibration (AOI 1-4) and validation (AOI 5-8) transect data

## Common Commands

All commands run from the `v2/` directory:

```bash
# Install package (editable)
pip install -e .

# Install all dependencies including geospatial
pip install -r requirements.txt

# Data preparation
python scripts/01_prepare_data.py --config config/default_config.yaml

# Training
python scripts/02_train_model.py --config config/default_config.yaml

# Evaluation
python scripts/03_evaluate_model.py --checkpoint experiments/runs/checkpoints/best_model.pth

# Inference on new AOI
python scripts/04_predict.py --checkpoint experiments/runs/checkpoints/best_model.pth --aoi_path ../../datasets/validation/aoi5

# Run tests
pytest tests/
```

## Architecture

### v2 Package Structure (`v2/cliff_dl/`)

```
cliff_dl/
├── data/
│   ├── preprocessing.py   # Feature engineering (13 features per point)
│   ├── dataset.py         # PyTorch Dataset for transects
│   └── loaders.py         # DataLoader with variable-length padding
├── models/
│   ├── cnn_lstm.py        # Hybrid 1D-CNN + BiLSTM + Attention architecture
│   ├── losses.py          # Multi-task loss (focal + smooth L1 + BCE)
│   └── metrics.py         # MAE, RMSE, R² metrics
├── training/
│   └── trainer.py         # Training loop with early stopping
└── inference/
    ├── postprocess.py     # Outlier removal, alongshore smoothing
    └── export.py          # Shapefile export for GIS
```

### Model Architecture

Input → 1D-CNN (3 layers) → BiLSTM (2 layers, bidirectional) → Multi-Head Attention (8 heads) → Multi-task outputs:
- Segmentation: [background, cliff_base, cliff_top] probabilities
- Regression: Distance offsets to base/top
- Confidence: Has-cliff probability

### Feature Engineering (13 dimensions per point)

Geometric (6): elevation_normalized, distance_normalized, elevation_gradient, elevation_curvature, plus raw values
Domain from v1 (4): seaward_slope, landward_slope, trendline_deviation, local_slope_change
Geomorphological (3): convexity_index, relative_elevation, slope_strength

## Configuration

Edit `v2/config/default_config.yaml` for hyperparameters. Key settings:

- `data.train_aois`: [1, 2, 3] - training AOIs
- `data.val_aois`: [4] - validation AOI
- `data.test_aois`: [5, 6, 7, 8] - held-out test AOIs
- `features.n_vert`: 20 - local slope window (must match v1.0)
- `loss.class_weights`: [1.0, 200.0, 200.0] - handles extreme class imbalance

## Data Format

Input files per AOI (`datasets/calibration/aoi1/`):
- `aoi1_points.txt` - CSV with FID, TransectID, Elevation (RASTERVALU), Distance (NEAR_DIST)
- `aoi1_base_true.shp` - Ground truth cliff base positions
- `aoi1_top_true.shp` - Ground truth cliff top positions

Elevation value of -9999 indicates nodata (interpolated during preprocessing).

## Key Constraints

- Cliff base must be below cliff top (enforced in postprocessing)
- `max_base_elevation`: 15m - bases above this are penalized
- `max_cliff_width`: 150m - unreasonable widths are penalized
- `min_base_top_distance`: 5m - minimum separation between base and top
