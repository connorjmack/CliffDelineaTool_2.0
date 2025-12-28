#!/usr/bin/env python3
"""
Visualize transect predictions vs ground truth.
Creates a GIF showing elevation profiles with predicted and actual cliff positions.
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import geopandas as gpd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_original_transect_data(aoi_name, data_root="../../datasets"):
    """Load original elevation data for an AOI."""
    aoi_path = Path(data_root) / "validation" / aoi_name

    # Load points
    points_file = aoi_path / f"{aoi_name}_points.txt"
    points = pd.read_csv(points_file)

    # Standardize column names (case-insensitive)
    col_map = {}
    for col in points.columns:
        col_upper = col.upper()
        if col_upper == 'FID' and 'PointID' not in points.columns:
            col_map[col] = 'PointID'
        elif col_upper in ['ID_1', 'OBJECTID'] and 'TransectID' not in points.columns:
            col_map[col] = 'TransectID'
        elif col_upper == 'RASTERVALU' and 'Elevation' not in points.columns:
            col_map[col] = 'Elevation'
        elif col_upper == 'NEAR_DIST' and 'Distance' not in points.columns:
            col_map[col] = 'Distance'

    if col_map:
        points.rename(columns=col_map, inplace=True)

    return points


def plot_transect(transect_data, predictions_row, ax, transect_idx):
    """
    Plot a single transect with predictions and ground truth.

    Args:
        transect_data: Dictionary with preprocessed transect data
        predictions_row: Row from predictions CSV
        ax: Matplotlib axis to plot on
        transect_idx: Index of transect for title
    """
    # Extract data
    distances = transect_data['distances']
    elev_normalized = transect_data['features'][:, 0]  # Normalized elevation

    # Denormalize elevation (approximate - multiply by 100 for realistic scale)
    # Better: we'll try to load original data
    elevation = elev_normalized * 100  # Rough approximation

    # Ground truth positions
    base_gt = transect_data['base_dist_gt']
    top_gt = transect_data['top_dist_gt']

    # Predicted positions
    base_pred = predictions_row['base_distance']
    top_pred = predictions_row['top_distance']

    # Plot elevation profile
    ax.plot(distances, elevation, 'k-', linewidth=1.5, label='Elevation Profile', alpha=0.7)
    ax.fill_between(distances, 0, elevation, alpha=0.1, color='gray')

    # Mark ground truth positions
    ax.axvline(base_gt, color='blue', linestyle='--', linewidth=2,
               label=f'GT Base ({base_gt:.1f}m)', alpha=0.7)
    ax.axvline(top_gt, color='green', linestyle='--', linewidth=2,
               label=f'GT Top ({top_gt:.1f}m)', alpha=0.7)

    # Mark predicted positions
    ax.axvline(base_pred, color='red', linestyle='-', linewidth=2,
               label=f'Pred Base ({base_pred:.1f}m)', alpha=0.8)
    ax.axvline(top_pred, color='orange', linestyle='-', linewidth=2,
               label=f'Pred Top ({top_pred:.1f}m)', alpha=0.8)

    # Highlight the cliff region
    if base_gt < top_gt:
        ax.axvspan(base_gt, top_gt, alpha=0.15, color='green', label='GT Cliff Zone')

    # Labels and formatting
    ax.set_xlabel('Distance from Sea (m)', fontsize=11)
    ax.set_ylabel('Elevation (m)', fontsize=11)
    ax.set_title(f'Transect #{transect_idx} (ID: {transect_data["transect_id"]})\n'
                 f'Base Error: {abs(base_pred - base_gt):.2f}m | '
                 f'Top Error: {abs(top_pred - top_gt):.2f}m',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(distances.min() - 5, distances.max() + 5)


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
        print(f"Warning: Could not load original data for transect {transect_id}: {e}")
        return None


def main():
    print("="*80)
    print("CliffDelineaTool v2.0 - Transect Visualization")
    print("="*80)

    # Paths
    test_data_path = Path("../data/preprocessed/test.pt")
    predictions_path = Path("../outputs/evaluation/predictions.csv")
    output_dir = Path("../outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    test_data = torch.load(test_data_path, weights_only=False)
    predictions = pd.read_csv(predictions_path)

    print(f"  Test transects: {len(test_data)}")
    print(f"  Predictions: {len(predictions)}")

    # Determine which transects to plot (every 5th, up to 500)
    max_idx = min(500, len(test_data))
    transect_indices = list(range(0, max_idx, 5))
    print(f"\n  Plotting transects: {transect_indices[:10]}... (total: {len(transect_indices)})")

    # Create individual frames
    frames = []

    for i, idx in enumerate(transect_indices):
        print(f"\r  Generating frame {i+1}/{len(transect_indices)}...", end='', flush=True)

        # Get transect data
        transect_data = test_data[idx]
        transect_id = transect_data['transect_id']

        # Find corresponding prediction
        pred_row = predictions[predictions['TransectID'] == transect_id]
        if len(pred_row) == 0:
            print(f"\n  Warning: No prediction found for transect {transect_id}, skipping")
            continue
        pred_row = pred_row.iloc[0]

        # Try to load original elevation data
        # Map test indices to AOI names (test set is AOIs 5-8)
        # This is approximate - we'll use the normalized elevation if we can't load
        original_data = None
        for aoi_num in [5, 6, 7, 8]:
            aoi_name = f"aoi{aoi_num}"
            original_data = load_original_elevation(transect_id, aoi_name)
            if original_data is not None:
                break

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        if original_data is not None and len(original_data) > 0:
            # Use original elevation data
            distances = original_data[:, 0]
            elevation = original_data[:, 1]

            # Plot elevation profile
            ax.plot(distances, elevation, 'k-', linewidth=1.5, label='Elevation Profile', alpha=0.7)
            ax.fill_between(distances, elevation.min() - 2, elevation, alpha=0.1, color='gray')
        else:
            # Use normalized elevation (scaled)
            distances = transect_data['distances']
            elev_norm = transect_data['features'][:, 0]
            # Scale to approximate real elevation (0-1 -> 0-50m typically)
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
        ymin, ymax = ax.get_ylim()

        # Ground truth
        ax.axvline(base_gt, color='blue', linestyle='--', linewidth=2.5,
                   label=f'GT Base ({base_gt:.1f}m)', alpha=0.7, zorder=5)
        ax.axvline(top_gt, color='green', linestyle='--', linewidth=2.5,
                   label=f'GT Top ({top_gt:.1f}m)', alpha=0.7, zorder=5)

        # Predictions
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

        # Save frame
        frame_path = output_dir / f"frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()

        frames.append(frame_path)

    print("\n\nCreating GIF...")

    # Create GIF
    images = [Image.open(f) for f in frames]
    gif_path = output_dir / "transects_animation.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1000,  # 1 second per frame
        loop=0
    )

    print(f"\n✓ GIF saved to: {gif_path}")
    print(f"  Frames: {len(frames)}")
    print(f"  Size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Clean up individual frames (optional)
    print("\nCleaning up individual frames...")
    for frame in frames:
        frame.unlink()

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
