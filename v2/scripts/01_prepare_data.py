#!/usr/bin/env python3
"""
Data Preparation Script for CliffDelineaTool v2.0

Converts raw shapefile data to preprocessed PyTorch tensors ready for training.

Usage:
    python 01_prepare_data.py --config ../config/default_config.yaml
    python 01_prepare_data.py --data_root ../datasets --output_dir ./data
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cliff_dl.data.dataset import create_dataset_from_aois


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cliff detection data for deep learning"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Root directory of datasets (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/preprocessed',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--use_soft_labels',
        action='store_true',
        default=True,
        help='Use soft (Gaussian) labels instead of hard labels'
    )
    parser.add_argument(
        '--seg_tolerance',
        type=float,
        default=None,
        help='Segmentation label tolerance in meters (overrides config)'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    data_root = args.data_root if args.data_root else config['data']['data_root']
    train_aois = config['data']['train_aois']
    val_aois = config['data']['val_aois']
    test_aois = config['data']['test_aois']
    n_vert = config['features']['n_vert']
    seg_tolerance = args.seg_tolerance if args.seg_tolerance else config['features']['seg_tolerance']

    print("=" * 80)
    print("CliffDelineaTool v2.0 - Data Preparation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  n_vert (slope window): {n_vert}")
    print(f"  Segmentation tolerance: {seg_tolerance} meters")
    print(f"  Use soft labels: {args.use_soft_labels}")
    print(f"\nDataset splits:")
    print(f"  Train AOIs: {train_aois}")
    print(f"  Validation AOIs: {val_aois}")
    print(f"  Test AOIs: {test_aois}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training data
    print("-" * 80)
    print("Preparing TRAINING data...")
    print("-" * 80)
    create_dataset_from_aois(
        data_root=data_root,
        aoi_list=train_aois,
        output_path=output_dir / 'train.pt',
        dataset_type='calibration',
        n_vert=n_vert,
        seg_tolerance=seg_tolerance,
        use_soft_labels=args.use_soft_labels
    )

    # Prepare validation data
    print("\n" + "-" * 80)
    print("Preparing VALIDATION data...")
    print("-" * 80)
    create_dataset_from_aois(
        data_root=data_root,
        aoi_list=val_aois,
        output_path=output_dir / 'val.pt',
        dataset_type='calibration',
        n_vert=n_vert,
        seg_tolerance=seg_tolerance,
        use_soft_labels=args.use_soft_labels
    )

    # Prepare test data
    print("\n" + "-" * 80)
    print("Preparing TEST data...")
    print("-" * 80)
    create_dataset_from_aois(
        data_root=data_root,
        aoi_list=test_aois,
        output_path=output_dir / 'test.pt',
        dataset_type='validation',
        n_vert=n_vert,
        seg_tolerance=seg_tolerance,
        use_soft_labels=args.use_soft_labels
    )

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)
    print(f"\nPreprocessed data saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    print(f"  - train.pt (AOIs {train_aois})")
    print(f"  - val.pt (AOIs {val_aois})")
    print(f"  - test.pt (AOIs {test_aois})")
    print("\nNext steps:")
    print("  1. Run training: python scripts/02_train_model.py")
    print("  2. Evaluate model: python scripts/03_evaluate_model.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
