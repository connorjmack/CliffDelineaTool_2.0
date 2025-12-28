================================================================================
NEW MODEL RESULTS - Trained on ALL 8 AOIs (70/10/20 split)
================================================================================

These are the CORRECT results from the retrained model.

OVERALL PERFORMANCE (All 4 Validation AOIs: 776 transects):
  - Base MAE: 2.13m (was 7.82m - 73% improvement!)
  - Top MAE: 12.35m (was 87.80m - 86% improvement!)
  - Base RMSE: 4.87m (was 19.81m)
  - Top RMSE: 31.37m (was 138.09m)

AOI-SPECIFIC PERFORMANCE:
  - AOI 5: Base 0.54m, Top 1.53m
  - AOI 6: Base 1.48m, Top 4.11m
  - AOI 7: Base 3.25m, Top 3.36m
  - AOI 8 (Point Conception): Base 3.31m, Top 36.26m

KEY IMPROVEMENT - AOI 8:
  OLD Model: Top MAE 189.63m (finding slope top instead of cliff top)
  NEW Model: Top MAE 36.26m (81% IMPROVEMENT!)

Note: Some transects in AOI8 still have large errors (transects 119-128),
but the overall performance improved dramatically from 189m to 36m average.

FILES IN THIS FOLDER:
  - all_transects.gif: 100 sample predictions from all 4 AOIs
  - worst_transects.gif: 20 worst predictions
  - metrics_summary.png: Performance metrics charts by AOI
  - scatter_plots.png: Predicted vs ground truth scatter plots
  - worst_transects_info.csv: Details of the 20 worst predictions

The model was trained for 26 epochs with early stopping on validation AOIs
containing samples from ALL 8 geographic locations, ensuring geographic
diversity in training.
================================================================================

================================================================================
FIXED: all_transects.gif now shows ALL AOIs
================================================================================

The GIF has been regenerated with stratified sampling to include transects 
from all 4 validation AOIs proportionally:

  - aoi5: 21 transects (21.6%)
  - aoi6: 30 transects (30.9%)
  - aoi7: 19 transects (19.6%)
  - aoi8: 27 transects (27.8%)
  Total: 97 frames

This gives a complete picture of model performance across all geographic 
locations, including the challenging Point Conception cliffs in AOI8.

Fixed: December 28, 2024
================================================================================
