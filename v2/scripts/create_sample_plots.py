#!/usr/bin/env python3
"""
Create sample static plots of transects.
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_original_elevation(transect_id, aoi_name, data_root="../../datasets"):
    """Load original elevation profile for a transect."""
    try:
        aoi_path = Path(data_root) / "validation" / aoi_name
        points_file = aoi_path / f"{aoi_name}_points.txt"

        if not points_file.exists():
            return None

        points = pd.read_csv(points_file)

        # Standardize column names
        col_map = {}
        for col in points.columns:
            col_upper = col.upper()
            if col_upper in ['ID_1', 'OBJECTID']:
                col_map[col] = 'TransectID'
            elif col_upper == 'RASTERVALU':
                col_map[col] = 'Elevation'
            elif col_upper == 'NEAR_DIST':
                col_map[col] = 'Distance'

        if col_map:
            points.rename(columns=col_map, inplace=True)

        # Get transect data
        transect = points[points['TransectID'] == transect_id].copy()
        transect = transect.sort_values('Distance')

        # Clean invalid elevations
        transect.loc[transect['Elevation'] < -50, 'Elevation'] = np.nan
        transect['Elevation'] = transect['Elevation'].interpolate(method='linear')
        transect['Elevation'] = transect['Elevation'].ffill().bfill()

        return transect[['Distance', 'Elevation']].values

    except Exception as e:
        return None


def plot_transect_sample(idx, test_data, predictions, output_path):
    """Plot a single transect and save to file."""
    transect_data = test_data[idx]
    transect_id = transect_data['transect_id']

    # Find corresponding prediction
    pred_row = predictions[predictions['TransectID'] == transect_id]
    if len(pred_row) == 0:
        print(f"  Warning: No prediction found for transect {transect_id}")
        return False
    pred_row = pred_row.iloc[0]

    # Try to load original elevation data
    original_data = None
    for aoi_num in [5, 6, 7, 8]:
        aoi_name = f"aoi{aoi_num}"
        original_data = load_original_elevation(transect_id, aoi_name)
        if original_data is not None and len(original_data) > 0:
            break

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    if original_data is not None and len(original_data) > 0:
        # Use original elevation data
        distances = original_data[:, 0]
        elevation = original_data[:, 1]

        ax.plot(distances, elevation, 'k-', linewidth=1.5, label='Elevation Profile', alpha=0.7)
        ax.fill_between(distances, elevation.min() - 2, elevation, alpha=0.1, color='gray')
    else:
        # Use normalized elevation (scaled)
        distances = transect_data['distances']
        elev_norm = transect_data['features'][:, 0]
        elevation = elev_norm * 50

        ax.plot(distances, elevation, 'k-', linewidth=1.5, label='Elevation Profile (normalized)', alpha=0.7)
        ax.fill_between(distances, 0, elevation, alpha=0.1, color='gray')

    # Ground truth positions
    base_gt = transect_data['base_dist_gt']
    top_gt = transect_data['top_dist_gt']

    # Predicted positions
    base_pred = pred_row['base_distance']
    top_pred = pred_row['top_distance']

    # Calculate errors
    base_error = abs(base_pred - base_gt)
    top_error = abs(top_pred - top_gt)

    # Mark positions
    ax.axvline(base_gt, color='blue', linestyle='--', linewidth=2.5,
               label=f'GT Base ({base_gt:.1f}m)', alpha=0.7, zorder=5)
    ax.axvline(top_gt, color='green', linestyle='--', linewidth=2.5,
               label=f'GT Top ({top_gt:.1f}m)', alpha=0.7, zorder=5)

    ax.axvline(base_pred, color='red', linestyle='-', linewidth=2.5,
               label=f'Pred Base ({base_pred:.1f}m)', alpha=0.8, zorder=6)
    ax.axvline(top_pred, color='orange', linestyle='-', linewidth=2.5,
               label=f'Pred Top ({top_pred:.1f}m)', alpha=0.8, zorder=6)

    # Shade cliff zone
    if base_gt < top_gt:
        ax.axvspan(base_gt, top_gt, alpha=0.15, color='green',
                  label='Ground Truth Cliff Zone', zorder=1)

    # Add error annotations
    ax.text(0.02, 0.98, f'Base Error: {base_error:.2f}m\nTop Error: {top_error:.2f}m',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Labels
    ax.set_xlabel('Distance from Sea (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Elevation (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Transect #{idx} (ID: {transect_id})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def main():
    print("Creating sample transect plots...")

    # Paths
    test_data_path = Path("../data/preprocessed/test.pt")
    predictions_path = Path("../outputs/evaluation/predictions.csv")
    output_dir = Path("../outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_data = torch.load(test_data_path, weights_only=False)
    predictions = pd.read_csv(predictions_path)

    # Create sample plots for indices 0, 50, 100, 150
    sample_indices = [0, 50, 100, 150]

    for idx in sample_indices:
        output_path = output_dir / f"sample_transect_{idx:03d}.png"
        print(f"  Creating plot for transect {idx}...")
        success = plot_transect_sample(idx, test_data, predictions, output_path)
        if success:
            print(f"    ✓ Saved to {output_path}")

    print("\n✓ Sample plots created!")


if __name__ == "__main__":
    main()
