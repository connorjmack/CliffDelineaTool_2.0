# CliffDelineaTool v2.0
## Deep Learning for Coastal Cliff Detection

CliffDelineaTool v2.0 is a PyTorch-based deep learning implementation for identifying coastal cliff base and top positions from Digital Elevation Models (DEMs). This version uses a hybrid 1D-CNN + BiLSTM architecture to replace the hand-crafted rule-based approach of v1.0.

---

## Key Features

- **Deep Learning Architecture**: Hybrid 1D-CNN + BiLSTM + Multi-Head Attention
- **Multi-Task Learning**: Joint segmentation, regression, and confidence prediction
- **Handles Class Imbalance**: Focal loss with soft labels
- **Alongshore Consistency**: Post-processing with outlier removal
- **GIS Integration**: Direct shapefile export compatible with v1.0 workflow
- **No Manual Tuning**: Single model works across different coastal environments

---

## Architecture Overview

```
Input: 1D Elevation Transect [seq_len, 12 features]
  ↓
1D-CNN Feature Extractor (3 layers: 64→128→64 channels)
  ↓
BiLSTM Encoder (2 layers, hidden=128, bidirectional)
  ↓
Multi-Head Self-Attention (8 heads)
  ↓
Multi-Task Outputs:
  - Segmentation: [background, cliff_base, cliff_top] probabilities
  - Regression: Distance offsets to base/top
  - Confidence: Has cliff probability
```

### Features (12-dimensional per point)

**Geometric Features:**
1. Raw elevation (m)
2. Normalized elevation [0,1]
3. Cross-shore distance (m)
4. Normalized distance [0,1]
5. Elevation gradient (∂z/∂x)
6. Elevation curvature (∂²z/∂x²)

**Domain Features (from v1.0):**
7. Seaward slope (degrees)
8. Landward slope (degrees)
9. Trendline deviation
10. Local slope change
11. Convexity index
12. Relative elevation

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- GDAL/OGR (for geospatial operations)

### Install Dependencies

```bash
cd v2
pip install -r requirements.txt
```

### Install GDAL (if not already installed)

**macOS:**
```bash
brew install gdal
pip install gdal==$(gdal-config --version)
```

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install gdal==$(gdal-config --version)
```

---

## Quick Start

### 1. Prepare Data

Convert raw shapefiles to preprocessed PyTorch tensors:

```bash
cd scripts
python 01_prepare_data.py --config ../config/default_config.yaml
```

This creates:
- `data/preprocessed/train.pt` (AOI 1, 2, 3)
- `data/preprocessed/val.pt` (AOI 4)
- `data/preprocessed/test.pt` (AOI 5, 6, 7, 8)

### 2. Train Model

```bash
python 02_train_model.py --config ../config/default_config.yaml
```

Training outputs:
- Model checkpoints: `experiments/runs/checkpoints/`
- TensorBoard logs: `experiments/runs/logs/`
- Best model: `experiments/runs/checkpoints/best_model.pth`

### 3. Evaluate Model

```bash
python 03_evaluate_model.py \
    --checkpoint ../experiments/runs/checkpoints/best_model.pth \
    --test_data ../data/preprocessed/test.pt
```

### 4. Predict on New Data

```bash
python 04_predict.py \
    --checkpoint ../experiments/runs/checkpoints/best_model.pth \
    --aoi_path ../../datasets/validation/aoi5 \
    --output_dir ../outputs/predictions
```

---

## Project Structure

```
v2/
├── cliff_dl/                      # Main Python package
│   ├── data/
│   │   ├── preprocessing.py       # Feature engineering (12 features)
│   │   ├── dataset.py             # PyTorch Dataset
│   │   └── loaders.py             # DataLoader with padding
│   ├── models/
│   │   ├── cnn_lstm.py            # Hybrid CNN-BiLSTM architecture
│   │   ├── losses.py              # Multi-task loss (focal + smooth L1)
│   │   └── metrics.py             # MAE, RMSE, R², accuracy metrics
│   ├── training/
│   │   └── trainer.py             # Training loop
│   └── inference/
│       ├── postprocess.py         # Outlier removal + smoothing
│       └── export.py              # Shapefile export
│
├── scripts/
│   ├── 01_prepare_data.py         # Data preprocessing
│   ├── 02_train_model.py          # Training script
│   ├── 03_evaluate_model.py       # Evaluation script
│   └── 04_predict.py              # Inference script
│
├── config/
│   └── default_config.yaml        # Hyperparameters
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
└── README.md                      # This file
```

---

## Configuration

Edit `config/default_config.yaml` to customize:

### Model Architecture
```yaml
model:
  cnn:
    channels: [64, 128, 64]
    kernel_sizes: [5, 5, 3]
    dropout: 0.2
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
  attention:
    num_heads: 8
    dim: 256
```

### Training Parameters
```yaml
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
```

### Loss Weights
```yaml
loss:
  lambda_seg: 1.0      # Segmentation loss
  lambda_reg: 0.5      # Regression loss
  lambda_conf: 0.3     # Confidence loss
  class_weights: [1.0, 50.0, 50.0]  # [background, base, top]
  focal_gamma: 2.0
```

### Data Splits
```yaml
data:
  train_aois: [1, 2, 3]
  val_aois: [4]
  test_aois: [5, 6, 7, 8]
  data_root: "../datasets"
```

---

## Expected Performance

### v1.0 Baseline (from Swirad & Young 2022)
- **Validation MAE**: ~3-7m
- **Requires**: Manual parameter tuning per AOI

### v2.0 Target
- **Validation MAE**: ~2-4m (30-50% improvement)
- **Requires**: No manual tuning, single model for all AOIs
- **Inference**: <1 second per transect on GPU

### Success Criteria
- ✓ MAE < 4m on test set (AOI 5-8)
- ✓ Outperforms v1.0 by ≥20%
- ✓ No manual parameter tuning
- ✓ Compatible GIS output (shapefiles)

---

## Data Format

### Input Requirements

**Per AOI directory** (e.g., `datasets/calibration/aoi1/`):
```
aoi1/
├── aoi1_points.txt              # Point elevations along transects
├── aoi1_base_true.shp           # Ground truth cliff base
├── aoi1_top_true.shp            # Ground truth cliff top
├── aoi1_dem.tif                 # Digital Elevation Model (optional)
└── aoi1_transects.shp           # Transect polylines (optional)
```

**Points file format** (`aoi1_points.txt`):
```csv
FID,ID_1,RASTERVALU,NEAR_DIST
0,1,-9999.0,0.0
1,1,-9999.0,0.99
2,1,0.537,1.98
...
```

- `FID`: Point ID
- `ID_1`: Transect ID
- `RASTERVALU`: Elevation (m, -9999 = nodata)
- `NEAR_DIST`: Distance from seaward end (m)

### Output Format

Predictions are exported as shapefiles compatible with v1.0:
- `base_predicted.shp`: Cliff base positions
- `top_predicted.shp`: Cliff top positions

Each shapefile contains:
- `TransectID`: Transect identifier
- `Distance`: Predicted cross-shore distance (m)
- `Elevation`: Elevation at predicted position (m)
- `Confidence`: Model confidence score (optional)
- `geometry`: Point geometry (UTM coordinates)

---

## Training Tips

### Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir experiments/runs/logs
```

View at: http://localhost:6006

### Handling Overfitting

If validation loss increases while training loss decreases:
1. Increase dropout (cnn: 0.3, lstm: 0.4)
2. Increase weight decay (0.0005)
3. Enable data augmentation (noise + vertical shift)
4. Reduce model size (fewer LSTM layers)

### Improving Performance

If MAE is too high:
1. Increase model capacity (more channels, larger hidden size)
2. Adjust loss weights (increase lambda_seg)
3. Use soft labels with smaller sigma (1.5m instead of 2.0m)
4. Enable GP smoothing in post-processing

---

## Comparison with v1.0

| Feature | v1.0 | v2.0 (Deep Learning) |
|---------|------|----------------------|
| **Approach** | Hand-crafted slope thresholds + trendlines | Learned features (CNN-BiLSTM) |
| **Features** | 4 hand-designed (slopes, trendline deviation) | 12 features (6 geometric + 6 domain) |
| **Training** | Manual parameter tuning per AOI | End-to-end learning, single model |
| **Generalization** | Requires calibration per region | Works across multiple regions |
| **Performance** | MAE ~3-7m | MAE ~2-4m (target) |
| **Inference** | Fast (Python/MATLAB) | GPU-accelerated |
| **Uncertainty** | No confidence scores | Outputs confidence per prediction |
| **Post-processing** | Moving median + OLS outlier removal | Same + optional GP smoothing |

---

## Known Limitations

1. **Limited Training Data**: Only 704 calibration transects
   - Mitigation: Strong regularization, data augmentation

2. **Extreme Class Imbalance**: ~2 labels per 89-653 points
   - Mitigation: Focal loss, soft labels, multi-task learning

3. **1D Transect-Based**: Doesn't use full 2D DEM context
   - Future v3.0: Move to 2D semantic segmentation

4. **GPU Recommended**: CPU training is very slow (10-20x slower)

---

## Troubleshooting

### GDAL Installation Issues

If `pip install gdal` fails:
```bash
# Install system GDAL first
brew install gdal  # macOS
sudo apt-get install gdal-bin libgdal-dev  # Ubuntu

# Then install Python bindings
pip install --global-option=build_ext --global-option="-I/usr/include/gdal" gdal==$(gdal-config --version)
```

### CUDA Out of Memory

Reduce batch size in `config/default_config.yaml`:
```yaml
training:
  batch_size: 8  # or 4
```

### Slow Data Loading

Increase number of workers:
```yaml
training:
  num_workers: 4  # Set to number of CPU cores
```

---

## Citation

If you use CliffDelineaTool v2.0, please cite both:

**Original v1.0 Paper:**
```
Swirad, Z.M. and Young, A.P., 2022. CliffDelineaTool v1.2.0: an algorithm for
identifying coastal cliff base and top positions. Geoscientific Model Development,
15(4), pp.1499-1512. https://doi.org/10.5194/gmd-15-1499-2022
```

**v2.0 Deep Learning Implementation:**
```
[To be published - reference this repository for now]
https://github.com/yourusername/CliffDelineaTool_2.0
```

---

## License

[Same as v1.0 - specify license here]

---

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: [your email]

---

## Acknowledgments

- Original algorithm: Swirad & Young (2022)
- Deep learning implementation: [Your name]
- Training data: [Data source acknowledgments]
