#!/usr/bin/env python3
"""
Prediction Script for CliffDelineaTool v2.0

Run inference on new AOI data and export predictions as shapefiles.

Usage:
    # Predict on a single AOI
    python 04_predict.py --checkpoint ../experiments/runs/checkpoints/best_model.pth \
                         --aoi_path ../../datasets/validation/aoi5 \
                         --output_dir ../outputs/predictions

    # Predict on multiple AOIs
    python 04_predict.py --checkpoint best_model.pth \
                         --data_root ../../datasets/validation \
                         --aoi_list 5 6 7 8 \
                         --output_dir ../outputs/predictions
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cliff_dl.data.preprocessing import preprocess_transect
from cliff_dl.models.cnn_lstm import CNN_BiLSTM_CliffDetector
from cliff_dl.inference.postprocess import (
    predictions_to_dataframe,
    postprocess_predictions
)
from cliff_dl.inference.export import (
    export_to_shapefiles,
    export_to_csv,
    compare_with_ground_truth
)


def load_and_preprocess_aoi(aoi_path: Path, n_vert: int = 20):
    """
    Load and preprocess a single AOI.

    Args:
        aoi_path: Path to AOI directory
        n_vert: Window size for slope calculation

    Returns:
        List of preprocessed transect dictionaries
    """
    aoi_name = aoi_path.name
    print(f"\nProcessing {aoi_name}...")

    # Load point data
    points_file = aoi_path / f"{aoi_name}_points.txt"
    if not points_file.exists():
        raise FileNotFoundError(f"Points file not found: {points_file}")

    points = pd.read_csv(points_file)

    # Standardize column names
    if 'FID' in points.columns and 'PointID' not in points.columns:
        points.rename(columns={'FID': 'PointID'}, inplace=True)
    if 'ID_1' in points.columns and 'TransectID' not in points.columns:
        points.rename(columns={'ID_1': 'TransectID'}, inplace=True)
    if 'RASTERVALU' in points.columns and 'Elevation' not in points.columns:
        points.rename(columns={'RASTERVALU': 'Elevation'}, inplace=True)
    if 'NEAR_DIST' in points.columns and 'Distance' not in points.columns:
        points.rename(columns={'NEAR_DIST': 'Distance'}, inplace=True)

    # Process each transect
    transects = []
    transect_ids = sorted(points['TransectID'].unique())

    for transect_id in transect_ids:
        transect_points = points[points['TransectID'] == transect_id].copy()

        # Preprocess (no ground truth for inference)
        preprocessed = preprocess_transect(
            transect_points,
            base_dist=None,
            top_dist=None,
            n_vert=n_vert,
            use_soft_labels=False
        )

        transects.append(preprocessed)

    print(f"  Preprocessed {len(transects)} transects")
    return transects


def predict_on_transects(model, transects, device, confidence_threshold=0.5):
    """
    Run inference on preprocessed transects.

    Args:
        model: Trained model
        transects: List of preprocessed transect dictionaries
        device: Device to use
        confidence_threshold: Confidence threshold for predictions

    Returns:
        DataFrame with predictions
    """
    model.eval()

    all_transect_ids = []
    all_base_distances = []
    all_top_distances = []
    all_base_confidences = []
    all_top_confidences = []

    print("\nRunning inference...")
    with torch.no_grad():
        for transect in tqdm(transects):
            # Prepare input
            features = torch.FloatTensor(transect['features']).unsqueeze(0).to(device)
            distances = torch.FloatTensor(transect['distances']).unsqueeze(0).to(device)
            lengths = torch.LongTensor([len(transect['features'])])

            # Predict
            predictions = model.predict_cliff_positions(
                features, distances, mask=None, lengths=lengths,
                confidence_threshold=confidence_threshold
            )

            # Collect results
            all_transect_ids.append(transect['transect_id'])
            all_base_distances.append(predictions['base_distances'][0].item())
            all_top_distances.append(predictions['top_distances'][0].item())
            all_base_confidences.append(predictions['base_confidences'][0].item())
            all_top_confidences.append(predictions['top_confidences'][0].item())

    # Create dataframe
    predictions_df = predictions_to_dataframe(
        all_transect_ids,
        all_base_distances,
        all_top_distances,
        all_base_confidences,
        all_top_confidences
    )

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Predict cliff positions on new data"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--aoi_path',
        type=str,
        default=None,
        help='Path to single AOI directory (e.g., datasets/validation/aoi5)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Root directory containing multiple AOIs'
    )
    parser.add_argument(
        '--aoi_list',
        type=int,
        nargs='+',
        default=None,
        help='List of AOI numbers to process (e.g., 5 6 7 8)'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='validation',
        help='Dataset type: calibration or validation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../outputs/predictions',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--postprocess',
        action='store_true',
        default=True,
        help='Apply post-processing (outlier removal + smoothing)'
    )
    parser.add_argument(
        '--export_shapefiles',
        action='store_true',
        default=True,
        help='Export predictions as shapefiles'
    )
    parser.add_argument(
        '--compare_ground_truth',
        action='store_true',
        help='Compare with ground truth if available'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CliffDelineaTool v2.0 - Prediction")
    print("=" * 80)

    # Validate arguments
    if args.aoi_path is None and (args.data_root is None or args.aoi_list is None):
        parser.error("Either --aoi_path or both --data_root and --aoi_list must be provided")

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

    # Create model
    print("\nCreating model...")
    model = CNN_BiLSTM_CliffDetector(
        input_dim=config['features']['input_dim'],
        cnn_channels=config['model']['cnn']['channels'],
        cnn_kernel_sizes=config['model']['cnn']['kernel_sizes'],
        cnn_dropout=config['model']['cnn']['dropout'],
        lstm_hidden_size=config['model']['lstm']['hidden_size'],
        lstm_num_layers=config['model']['lstm']['num_layers'],
        lstm_dropout=config['model']['lstm']['dropout'],
        attention_heads=config['model']['attention']['num_heads'],
        attention_dim=config['model']['attention']['dim'],
        attention_dropout=config['model']['attention']['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get list of AOI paths
    if args.aoi_path:
        aoi_paths = [Path(args.aoi_path)]
    else:
        data_root = Path(args.data_root)
        aoi_paths = [data_root / args.dataset_type / f"aoi{num}" for num in args.aoi_list]

    # Process each AOI
    for aoi_path in aoi_paths:
        if not aoi_path.exists():
            print(f"\nWarning: AOI path not found: {aoi_path}, skipping")
            continue

        print("\n" + "-" * 80)
        print(f"Processing {aoi_path.name}")
        print("-" * 80)

        # Load and preprocess
        transects = load_and_preprocess_aoi(aoi_path, n_vert=config['features']['n_vert'])

        # Predict
        predictions_df = predict_on_transects(
            model, transects, device,
            confidence_threshold=config['postprocess']['confidence_threshold']
        )

        print(f"\nPredictions summary:")
        valid_count = ((predictions_df['base_distance'] > 0) &
                       (predictions_df['top_distance'] > 0)).sum()
        print(f"  Total transects: {len(predictions_df)}")
        print(f"  Valid predictions: {valid_count}")
        print(f"  Invalid/rejected: {len(predictions_df) - valid_count}")

        # Apply post-processing
        if args.postprocess:
            print("\nApplying post-processing...")
            predictions_df = postprocess_predictions(
                predictions_df,
                smooth_window=config['postprocess']['smooth_window'],
                outlier_threshold=config['postprocess']['outlier_std_threshold'],
                min_base_top_distance=config['postprocess']['min_base_top_distance']
            )

        # Save predictions
        output_dir = Path(args.output_dir) / aoi_path.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export CSV
        csv_path = output_dir / 'predictions.csv'
        export_to_csv(predictions_df, csv_path)

        # Export shapefiles
        if args.export_shapefiles:
            points_shapefile = aoi_path / f"{aoi_path.name}_points.shp"
            if points_shapefile.exists():
                print("\nExporting shapefiles...")
                export_to_shapefiles(
                    predictions_df,
                    str(points_shapefile),
                    str(output_dir),
                    base_filename='base_predicted',
                    top_filename='top_predicted'
                )
            else:
                print(f"Warning: Points shapefile not found: {points_shapefile}")
                print("  Skipping shapefile export")

        # Compare with ground truth if requested
        if args.compare_ground_truth:
            base_true = aoi_path / f"{aoi_path.name}_base_true.shp"
            top_true = aoi_path / f"{aoi_path.name}_top_true.shp"

            if base_true.exists() and top_true.exists():
                print("\nComparing with ground truth...")
                comparison_df = compare_with_ground_truth(
                    predictions_df,
                    str(base_true),
                    str(top_true),
                    output_path=str(output_dir / 'comparison.csv')
                )

                # Print comparison metrics
                valid_base = comparison_df['base_abs_error'].notna()
                valid_top = comparison_df['top_abs_error'].notna()

                if valid_base.any():
                    base_mae = comparison_df.loc[valid_base, 'base_abs_error'].mean()
                    print(f"  Base MAE: {base_mae:.2f} m")

                if valid_top.any():
                    top_mae = comparison_df.loc[valid_top, 'top_abs_error'].mean()
                    print(f"  Top MAE: {top_mae:.2f} m")
            else:
                print("\nWarning: Ground truth files not found, skipping comparison")

        print(f"\nResults saved to: {output_dir.absolute()}")

    print("\n" + "=" * 80)
    print("Prediction complete!")
    print("=" * 80)
    print(f"\nOutput directory: {Path(args.output_dir).absolute()}")
    print("\nGenerated files:")
    print("  - predictions.csv: Predicted cliff positions")
    if args.export_shapefiles:
        print("  - base_predicted.shp: Cliff base shapefile")
        print("  - top_predicted.shp: Cliff top shapefile")
    if args.compare_ground_truth:
        print("  - comparison.csv: Predictions vs ground truth")
    print("=" * 80)


if __name__ == '__main__':
    main()
