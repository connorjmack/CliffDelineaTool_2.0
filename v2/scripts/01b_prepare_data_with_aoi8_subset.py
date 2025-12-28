#!/usr/bin/env python3
"""
Data Preparation Script with AOI8 Subset
Adds a portion of AOI8 to training to expose model to Point Conception morphology
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cliff_dl.data.dataset import create_dataset_from_aois, load_aoi_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cliff detection data with AOI8 subset in training"
    )
    parser.add_argument(
        '--aoi8_fraction',
        type=float,
        default=0.35,
        help='Fraction of AOI8 to include in training (default: 0.35 = ~75 transects)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for AOI8 split'
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Configuration
    data_root = Path("../../datasets")
    output_dir = Path("../data/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_vert = 20
    seg_tolerance = 2.0
    use_soft_labels = True

    print("=" * 80)
    print("CliffDelineaTool v2.0 - Data Preparation (with AOI8 subset)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  Output directory: {output_dir}")
    print(f"  n_vert: {n_vert}")
    print(f"  Segmentation tolerance: {seg_tolerance}")
    print(f"  Use soft labels: {use_soft_labels}")
    print(f"  AOI8 training fraction: {args.aoi8_fraction}")
    print(f"  Random seed: {args.seed}")

    # Prepare base training data (AOIs 1-3)
    print("\n" + "-" * 80)
    print("Preparing BASE TRAINING data (AOIs 1-3)...")
    print("-" * 80)
    base_transects = []
    for aoi_num in [1, 2, 3]:
        aoi_path = data_root / 'calibration' / f"aoi{aoi_num}"
        transects = load_aoi_data(aoi_path, n_vert, seg_tolerance, use_soft_labels)
        base_transects.extend(transects)

    print(f"Base training transects: {len(base_transects)}")

    # Load ALL of AOI8
    print("\n" + "-" * 80)
    print("Loading AOI8 (Point Conception)...")
    print("-" * 80)
    aoi8_path = data_root / 'validation' / 'aoi8'
    aoi8_all_transects = load_aoi_data(aoi8_path, n_vert, seg_tolerance, use_soft_labels)

    # Split AOI8 into train/test
    n_aoi8_train = int(len(aoi8_all_transects) * args.aoi8_fraction)
    n_aoi8_test = len(aoi8_all_transects) - n_aoi8_train

    print(f"\nAOI8 split:")
    print(f"  Total AOI8 transects: {len(aoi8_all_transects)}")
    print(f"  Train portion: {n_aoi8_train} ({args.aoi8_fraction*100:.1f}%)")
    print(f"  Test portion: {n_aoi8_test} ({(1-args.aoi8_fraction)*100:.1f}%)")

    # Shuffle and split
    random.shuffle(aoi8_all_transects)
    aoi8_train = aoi8_all_transects[:n_aoi8_train]
    aoi8_test = aoi8_all_transects[n_aoi8_train:]

    # Combine for final training set
    train_transects = base_transects + aoi8_train
    print(f"\nFinal training set: {len(train_transects)} transects")
    print(f"  AOIs 1-3: {len(base_transects)}")
    print(f"  AOI8 subset: {len(aoi8_train)}")

    # Save training data
    train_path = output_dir / 'train.pt'
    torch.save(train_transects, train_path)
    print(f"\n✓ Saved training data to {train_path}")
    print(f"  File size: {train_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Prepare validation data (AOI 4)
    print("\n" + "-" * 80)
    print("Preparing VALIDATION data (AOI 4)...")
    print("-" * 80)
    aoi4_path = data_root / 'calibration' / 'aoi4'
    val_transects = load_aoi_data(aoi4_path, n_vert, seg_tolerance, use_soft_labels)

    val_path = output_dir / 'val.pt'
    torch.save(val_transects, val_path)
    print(f"\n✓ Saved validation data to {val_path}")
    print(f"  File size: {val_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Prepare test data (AOIs 5, 6, 7, and remaining AOI8)
    print("\n" + "-" * 80)
    print("Preparing TEST data (AOIs 5, 6, 7, 8)...")
    print("-" * 80)
    test_transects = []

    # Load AOIs 5, 6, 7
    for aoi_num in [5, 6, 7]:
        aoi_path = data_root / 'validation' / f"aoi{aoi_num}"
        transects = load_aoi_data(aoi_path, n_vert, seg_tolerance, use_soft_labels)
        test_transects.extend(transects)

    # Add remaining AOI8
    test_transects.extend(aoi8_test)

    print(f"\nTest set breakdown:")
    print(f"  AOIs 5, 6, 7: {len(test_transects) - len(aoi8_test)}")
    print(f"  AOI8 (test portion): {len(aoi8_test)}")
    print(f"  Total: {len(test_transects)}")

    test_path = output_dir / 'test.pt'
    torch.save(test_transects, test_path)
    print(f"\n✓ Saved test data to {test_path}")
    print(f"  File size: {test_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save AOI8 split info
    split_info = {
        'total_aoi8': len(aoi8_all_transects),
        'aoi8_train_count': len(aoi8_train),
        'aoi8_test_count': len(aoi8_test),
        'aoi8_train_ids': [t['transect_id'] for t in aoi8_train],
        'aoi8_test_ids': [t['transect_id'] for t in aoi8_test],
        'seed': args.seed,
        'fraction': args.aoi8_fraction
    }

    import json
    split_info_path = output_dir / 'aoi8_split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\n✓ Saved AOI8 split info to {split_info_path}")

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)
    print(f"\nDataset Summary:")
    print(f"  Training:   {len(train_transects):4d} transects (AOIs 1-3 + {args.aoi8_fraction*100:.0f}% of AOI8)")
    print(f"  Validation: {len(val_transects):4d} transects (AOI 4)")
    print(f"  Test:       {len(test_transects):4d} transects (AOIs 5-7 + {(1-args.aoi8_fraction)*100:.0f}% of AOI8)")
    print(f"\nNext steps:")
    print(f"  1. Train: python scripts/02_train_model.py")
    print(f"  2. Evaluate: python scripts/03_evaluate_model.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
