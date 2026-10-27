# CliffDelineaTool v2.0

A PyTorch model that finds the coastal cliff base (toe) and top on cross-shore elevation transects. It takes the same input as v1 (Swirad & Young, 2022) but learns the positions from labelled transects instead of using thresholds set for each site.

## Install

Python 3.11 or later.

```bash
cd v2
pip install -r requirements.txt   # exact environment the published model was trained in
# or: pip install -e .            # the package with compatible version ranges
```

## Run

From `v2/scripts/`; add `--device cpu` without a CUDA GPU.

```bash
python 01c_prepare_data_proper_split.py   # AOIs 1-8 -> ../data/preprocessed/{train,val,test}.pt
python 02_train_model.py                  # -> experiments/runs/checkpoints/best_model.pth
python 03_evaluate_model.py --checkpoint experiments/runs/checkpoints/best_model.pth
python 04_predict.py --checkpoint experiments/runs/checkpoints/best_model.pth --aoi_path ../../datasets/validation/aoi5
```

`02_train_model.py` writes to `experiments/runs/`, the folder that holds the published weights. Copy them elsewhere first, or change `paths` in `config/default_config.yaml`.

To run the model on your own DEMs, use `sd_county_.py --pair TRANSECTS.shp DEM.tif --checkpoint ... --output-dir ...`. It samples each transect from the DEM and writes predictions, base and top shapefiles, and a GIF.

## Input and output

Each AOI folder follows `datasets/` (see `datasets/readme.md`):
- `aoiX_points.txt` is the input. It is a CSV with one row per point: `FID`, `ID_1` (transect ID), `RASTERVALU` (elevation in m, −9999 for no data) and `NEAR_DIST` (distance from the seaward end in m).
- `aoiX_base_true.shp` and `aoiX_top_true.shp` are needed only for training and evaluation.
- `aoiX_points.shp` is needed only for shapefile export.

`04_predict.py` writes:
- `predictions.csv`: per transect, the base and top distance and confidence, plus post-processed `*_smoothed` columns.
- `base_predicted.shp` and `top_predicted.shp`: the raw positions, with `TransectID`, `Distance` and `Elevation`.
- `comparison.csv`: written only with `--compare_ground_truth`.

## Model

- **Features:** 13 per point, computed in `cliff_dl/data/preprocessing.compute_features`:
  - normalised elevation and distance, gradient and curvature
  - v1's seaward and landward slopes (`n_vert` = 20), trendline deviation, slope change and convexity
  - relative elevation, a below-15 m flag, shore proximity and maximum local slope
- **Network:** a 3-layer 1D CNN (64-128-64 channels), then a 2-layer bidirectional LSTM (hidden size 128), then 8-head self-attention.
- **Outputs:** three heads give per-point class (background, base, top), per-point distance offsets and a has-cliff confidence.
- **Loss:** focal loss plus smooth L1, a confidence term and elevation and width penalties.
- **Settings:** `config/default_config.yaml`.

## Published model

| | |
|---|---|
| Weights | `scripts/experiments/runs/checkpoints/best_model.pth`, sha256 `e2ea574661062a6a7b71b3c703605789988afd33c1bdfe5e624279bba117caf1`. Not in git; to be archived on Zenodo with v2.0.0. |
| Training | Epoch 10 (lowest validation loss) of a 26-epoch run |
| Split | `01c`: each of AOIs 1-8 split 70/10/20 by transect, seed 42, giving 1,032 / 146 / 302 transects |

Test-set error (302 transects, `03_evaluate_model.py`, raw positions):

| | MAE | RMSE | Within 2 m | Within 5 m |
|---|---|---|---|---|
| Base | 2.0 m | 4.0 m | 81 % | 91 % |
| Top | 9.5 m | 27.3 m | 68 % | 81 % |

## Known limitations

- **Within-site test only:** the test transects come from the same eight AOIs as the training transects, so these scores do not show how the model does at a new site. A leave-one-AOI-out test has not been run.
- **Post-processing hurts:** the outlier removal and smoothing in `cliff_dl/inference/postprocess.py` makes test error much worse (base MAE 25.5 m, top 73.1 m). Use the raw positions, which are what the shapefiles contain.
- **No automated tests:** the repository has none yet.

## Other scripts

- `compare_v1_v2.py` and `run_v1_validation.py` run v1 on AOIs 5-8 and compare it with v2.
- `create_aoi_gifs.py`, `create_output_figures.py`, `create_sample_plots.py` and `visualize_transects.py` make figures and GIFs.
- `01_prepare_data.py` and `01b_prepare_data_with_aoi8_subset.py` are earlier splits, kept for the record.

For citation and licence, see the [root README](../README.md).
