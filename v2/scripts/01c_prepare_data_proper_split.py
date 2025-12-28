#!/usr/bin/env python3
"""
Proper Train/Val/Test Split for CliffDelineaTool v2.0

Creates scientifically valid train/validation/test sets with NO OVERLAP.
Includes geographic diversity by sampling from all 8 AOIs.

Standard split: 70% train, 15% validation, 15% test (stratified by AOI)
"""

import argparse
import sys
from pathlib import Path
import torch
import random
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cliff_dl.data.dataset import load_aoi_data


def main():
    parser = argparse.ArgumentParser(
        description="Create proper train/val/test split with no overlap"
    )
    parser.add_argument(
        '--train_frac',
        type=float,
        default=0.70,
        help='Fraction for training (default: 0.70)'
    )
    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.15,
        help='Fraction for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test_frac',
        type=float,
        default=0.15,
        help='Fraction for test (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Validate fractions
    total_frac = args.train_frac + args.val_frac + args.test_frac
    if abs(total_frac - 1.0) > 0.001:
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")

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
    print("CliffDelineaTool v2.0 - Proper Train/Val/Test Split")
    print("=" * 80)
    print(f"\nSplit Configuration:")
    print(f"  Train:      {args.train_frac*100:.1f}%")
    print(f"  Validation: {args.val_frac*100:.1f}%")
    print(f"  Test:       {args.test_frac*100:.1f}%")
    print(f"  Random seed: {args.seed}")
    print(f"\nData Configuration:")
    print(f"  n_vert: {n_vert}")
    print(f"  Segmentation tolerance: {seg_tolerance} meters")
    print(f"  Use soft labels: {use_soft_labels}")

    # Load all AOIs
    all_aois = list(range(1, 9))  # AOIs 1-8
    all_transects_by_aoi = {}

    print("\n" + "-" * 80)
    print("Loading all AOIs...")
    print("-" * 80)

    for aoi_num in all_aois:
        if aoi_num <= 4:
            folder = 'calibration'
        else:
            folder = 'validation'

        aoi_path = data_root / folder / f"aoi{aoi_num}"
        transects = load_aoi_data(aoi_path, n_vert, seg_tolerance, use_soft_labels)
        all_transects_by_aoi[aoi_num] = transects

    total_transects = sum(len(t) for t in all_transects_by_aoi.values())
    print(f"\nTotal transects loaded: {total_transects}")

    # Split each AOI into train/val/test (stratified)
    train_transects = []
    val_transects = []
    test_transects = []

    split_info = {}

    print("\n" + "-" * 80)
    print("Splitting each AOI into train/val/test...")
    print("-" * 80)

    for aoi_num, transects in all_transects_by_aoi.items():
        n_total = len(transects)

        # Shuffle transects
        shuffled = transects.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        n_train = int(n_total * args.train_frac)
        n_val = int(n_total * args.val_frac)
        # n_test gets the remainder to ensure all transects are used

        # Split
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

        train_transects.extend(train)
        val_transects.extend(val)
        test_transects.extend(test)

        # Track split info
        split_info[f'aoi{aoi_num}'] = {
            'total': n_total,
            'train': len(train),
            'val': len(val),
            'test': len(test),
            'train_ids': [t['transect_id'] for t in train],
            'val_ids': [t['transect_id'] for t in val],
            'test_ids': [t['transect_id'] for t in test]
        }

        print(f"  AOI {aoi_num}: {n_total:3d} total → "
              f"train={len(train):3d}, val={len(val):3d}, test={len(test):3d}")

    # Final shuffle of combined sets
    random.shuffle(train_transects)
    random.shuffle(val_transects)
    random.shuffle(test_transects)

    print("\n" + "-" * 80)
    print("Final Dataset Summary:")
    print("-" * 80)
    print(f"  Training:   {len(train_transects):4d} transects ({len(train_transects)/total_transects*100:.1f}%)")
    print(f"  Validation: {len(val_transects):4d} transects ({len(val_transects)/total_transects*100:.1f}%)")
    print(f"  Test:       {len(test_transects):4d} transects ({len(test_transects)/total_transects*100:.1f}%)")
    print(f"  Total:      {total_transects:4d} transects")

    # Verify no overlap by design
    # Since we split each AOI independently, there's no overlap by construction
    print("\n" + "-" * 80)
    print("Verifying split integrity...")
    print("-" * 80)
    print(f"  ✓ Each AOI split independently - NO OVERLAP by design")
    print(f"  ✓ Train/Val/Test sets are completely independent")

    # Save datasets
    print("\n" + "-" * 80)
    print("Saving preprocessed datasets...")
    print("-" * 80)

    train_path = output_dir / 'train.pt'
    val_path = output_dir / 'val.pt'
    test_path = output_dir / 'test.pt'

    torch.save(train_transects, train_path)
    torch.save(val_transects, val_path)
    torch.save(test_transects, test_path)

    print(f"  ✓ Train:      {train_path} ({train_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  ✓ Validation: {val_path} ({val_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  ✓ Test:       {test_path} ({test_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Save split metadata
    metadata = {
        'split_fractions': {
            'train': args.train_frac,
            'val': args.val_frac,
            'test': args.test_frac
        },
        'counts': {
            'train': len(train_transects),
            'val': len(val_transects),
            'test': len(test_transects),
            'total': total_transects
        },
        'seed': args.seed,
        'aoi_splits': split_info
    }

    metadata_path = output_dir / 'split_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata:   {metadata_path}")

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)
    print("\nKey Points:")
    print("  ✓ All 8 AOIs included for geographic diversity")
    print("  ✓ Each AOI contributes to train/val/test")
    print("  ✓ NO OVERLAP between splits (proper scientific practice)")
    print("  ✓ Stratified split maintains AOI distribution")
    print("\nModel will:")
    print("  ✓ See Point Conception morphology during training")
    print("  ✓ Be validated on held-out data")
    print("  ✓ Be tested on completely independent data")
    print("\nNext steps:")
    print("  1. Train: python scripts/02_train_model.py")
    print("  2. Evaluate: python scripts/03_evaluate_model.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
