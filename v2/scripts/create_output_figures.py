#!/usr/bin/env python3
"""
Create comprehensive output figures for model evaluation.
Generates metrics plots, error distributions, and visualizations.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_all_comparisons(predictions_dir):
    """Load comparison data from all AOIs."""
    predictions_dir = Path(predictions_dir)
    all_data = []

    for aoi_dir in sorted(predictions_dir.iterdir()):
        if not aoi_dir.is_dir():
            continue

        comparison_file = aoi_dir / 'comparison.csv'
        if comparison_file.exists():
            df = pd.read_csv(comparison_file)
            df['AOI'] = aoi_dir.name
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    else:
        return None


def load_original_elevation(transect_id, aoi_name, data_root="../../datasets"):
    """Load original elevation profile for a transect."""
    try:
        aoi_path = Path(data_root) / "validation" / aoi_name
        points_file = aoi_path / f"{aoi_name}_points.txt"

        if not points_file.exists():
            return None

        points = pd.read_csv(points_file)

        # Standardize column names (case-insensitive)
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


def create_metrics_summary(df, output_dir):
    """Create summary metrics plots."""
    output_dir = Path(output_dir)

    # Calculate metrics by AOI
    aoi_metrics = []
    for aoi in df['AOI'].unique():
        aoi_df = df[df['AOI'] == aoi]
        aoi_metrics.append({
            'AOI': aoi,
            'Base MAE': aoi_df['base_abs_error'].mean(),
            'Top MAE': aoi_df['top_abs_error'].mean(),
            'Base RMSE': np.sqrt((aoi_df['base_error']**2).mean()),
            'Top RMSE': np.sqrt((aoi_df['top_error']**2).mean()),
            'Count': len(aoi_df)
        })

    metrics_df = pd.DataFrame(aoi_metrics)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # MAE by AOI
    ax = axes[0, 0]
    x = np.arange(len(metrics_df))
    width = 0.35
    ax.bar(x - width/2, metrics_df['Base MAE'], width, label='Base MAE', alpha=0.8)
    ax.bar(x + width/2, metrics_df['Top MAE'], width, label='Top MAE', alpha=0.8)
    ax.set_xlabel('AOI', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (m)', fontweight='bold', fontsize=12)
    ax.set_title('MAE by AOI', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['AOI'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE by AOI
    ax = axes[0, 1]
    ax.bar(x - width/2, metrics_df['Base RMSE'], width, label='Base RMSE', alpha=0.8)
    ax.bar(x + width/2, metrics_df['Top RMSE'], width, label='Top RMSE', alpha=0.8)
    ax.set_xlabel('AOI', fontweight='bold', fontsize=12)
    ax.set_ylabel('Root Mean Square Error (m)', fontweight='bold', fontsize=12)
    ax.set_title('RMSE by AOI', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['AOI'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution - Base
    ax = axes[1, 0]
    ax.hist(df['base_abs_error'].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(df['base_abs_error'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["base_abs_error"].mean():.2f}m')
    ax.axvline(df['base_abs_error'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {df["base_abs_error"].median():.2f}m')
    ax.set_xlabel('Absolute Error (m)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax.set_title('Base Error Distribution', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution - Top
    ax = axes[1, 1]
    ax.hist(df['top_abs_error'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(df['top_abs_error'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["top_abs_error"].mean():.2f}m')
    ax.axvline(df['top_abs_error'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {df["top_abs_error"].median():.2f}m')
    ax.set_xlabel('Absolute Error (m)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax.set_title('Top Error Distribution', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved metrics summary to {output_dir / 'metrics_summary.png'}")

    return metrics_df


def create_scatter_plots(df, output_dir):
    """Create scatter plots of predicted vs actual."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Base scatter
    ax = axes[0]
    valid_base = df[df['base_distance_true'].notna() & df['base_distance'].notna()]
    ax.scatter(valid_base['base_distance_true'], valid_base['base_distance'],
              alpha=0.5, s=30)

    # Perfect prediction line
    min_val = min(valid_base['base_distance_true'].min(), valid_base['base_distance'].min())
    max_val = max(valid_base['base_distance_true'].max(), valid_base['base_distance'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Ground Truth (m)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted (m)', fontweight='bold', fontsize=12)
    ax.set_title('Base Predictions vs Ground Truth', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Top scatter
    ax = axes[1]
    valid_top = df[df['top_distance_true'].notna() & df['top_distance'].notna()]
    ax.scatter(valid_top['top_distance_true'], valid_top['top_distance'],
              alpha=0.5, s=30, color='orange')

    min_val = min(valid_top['top_distance_true'].min(), valid_top['top_distance'].min())
    max_val = max(valid_top['top_distance_true'].max(), valid_top['top_distance'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Ground Truth (m)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted (m)', fontweight='bold', fontsize=12)
    ax.set_title('Top Predictions vs Ground Truth', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved scatter plots to {output_dir / 'scatter_plots.png'}")


def plot_transect_visualization(row, output_path, data_root="../../datasets"):
    """Create a single transect visualization."""
    transect_id = row['TransectID']
    aoi_name = row['AOI']

    # Load original elevation data
    original_data = load_original_elevation(transect_id, aoi_name, data_root)

    if original_data is None or len(original_data) == 0:
        return False

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    distances = original_data[:, 0]
    elevation = original_data[:, 1]

    # Plot elevation profile
    ax.plot(distances, elevation, 'k-', linewidth=1.5, label='Elevation Profile', alpha=0.7)
    ax.fill_between(distances, elevation.min() - 2, elevation, alpha=0.1, color='gray')

    # Ground truth positions
    base_gt = row['base_distance_true']
    top_gt = row['top_distance_true']

    # Predicted positions
    base_pred = row['base_distance']
    top_pred = row['top_distance']

    # Calculate errors
    base_error = abs(base_pred - base_gt) if pd.notna(base_gt) and pd.notna(base_pred) else np.nan
    top_error = abs(top_pred - top_gt) if pd.notna(top_gt) and pd.notna(top_pred) else np.nan

    # Calculate combined MAE for this transect
    mae = np.nanmean([base_error, top_error])

    # Mark positions
    if pd.notna(base_gt):
        ax.axvline(base_gt, color='blue', linestyle='--', linewidth=2.5,
                   label=f'GT Base ({base_gt:.1f}m)', alpha=0.7, zorder=5)
    if pd.notna(top_gt):
        ax.axvline(top_gt, color='green', linestyle='--', linewidth=2.5,
                   label=f'GT Top ({top_gt:.1f}m)', alpha=0.7, zorder=5)

    if pd.notna(base_pred):
        ax.axvline(base_pred, color='red', linestyle='-', linewidth=2.5,
                   label=f'Pred Base ({base_pred:.1f}m)', alpha=0.8, zorder=6)
    if pd.notna(top_pred):
        ax.axvline(top_pred, color='orange', linestyle='-', linewidth=2.5,
                   label=f'Pred Top ({top_pred:.1f}m)', alpha=0.8, zorder=6)

    # Shade cliff zone
    if pd.notna(base_gt) and pd.notna(top_gt) and base_gt < top_gt:
        ax.axvspan(base_gt, top_gt, alpha=0.15, color='green',
                  label='Ground Truth Cliff Zone', zorder=1)

    # Add error annotations
    error_text = ''
    if pd.notna(base_error):
        error_text += f'Base Error: {base_error:.2f}m\n'
    if pd.notna(top_error):
        error_text += f'Top Error: {top_error:.2f}m\n'
    if pd.notna(mae):
        error_text += f'Mean Error: {mae:.2f}m'

    ax.text(0.02, 0.98, error_text.strip(),
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Labels
    ax.set_xlabel('Distance from Sea (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Elevation (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'{aoi_name} - Transect ID: {transect_id}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return True


def create_transect_gif(df, output_dir, data_root="../../datasets", max_transects=100, step=1):
    """Create GIF of transect visualizations."""
    output_dir = Path(output_dir)
    frames_dir = output_dir / 'gif_frames'
    frames_dir.mkdir(exist_ok=True)

    print(f"\nCreating transect GIF (max {max_transects} transects, stratified by AOI)...")

    # Stratified sampling to get representative sample from all AOIs
    samples_per_aoi = []
    for aoi in sorted(df['AOI'].unique()):
        aoi_df = df[df['AOI'] == aoi]
        # Sample proportionally based on AOI size
        n_samples = max(1, int(max_transects * len(aoi_df) / len(df)))
        if len(aoi_df) <= n_samples:
            samples_per_aoi.append(aoi_df)
        else:
            samples_per_aoi.append(aoi_df.sample(n=n_samples, random_state=42))

    sampled_df = pd.concat(samples_per_aoi, ignore_index=True)
    # Ensure we don't exceed max_transects
    if len(sampled_df) > max_transects:
        sampled_df = sampled_df.sample(n=max_transects, random_state=42)

    print(f"  Selected {len(sampled_df)} transects:")
    for aoi in sorted(sampled_df['AOI'].unique()):
        count = len(sampled_df[sampled_df['AOI'] == aoi])
        print(f"    {aoi}: {count} transects")

    frames = []
    for i, (idx, row) in enumerate(tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Generating frames")):
        frame_path = frames_dir / f"frame_{i:04d}.png"

        success = plot_transect_visualization(row, frame_path, data_root)

        if success:
            frames.append(frame_path)

    if not frames:
        print("  Warning: No frames generated")
        return None

    print(f"  Generated {len(frames)} frames")
    print(f"  Creating GIF...")

    # Create GIF
    images = [Image.open(f) for f in frames]
    gif_path = output_dir / "all_transects.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=500,  # 0.5 second per frame
        loop=0
    )

    print(f"✓ Saved GIF to {gif_path}")
    print(f"  Size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Clean up frames
    print("  Cleaning up frames...")
    for frame in frames:
        frame.unlink()
    frames_dir.rmdir()

    return gif_path


def create_worst_transects_gif(df, output_dir, data_root="../../datasets", n_worst=20):
    """Create GIF of worst performing transects."""
    output_dir = Path(output_dir)
    frames_dir = output_dir / 'worst_frames'
    frames_dir.mkdir(exist_ok=True)

    print(f"\nCreating worst {n_worst} transects GIF...")

    # Calculate mean absolute error for each transect
    df['mean_abs_error'] = df[['base_abs_error', 'top_abs_error']].mean(axis=1)

    # Get worst transects
    worst_df = df.nlargest(n_worst, 'mean_abs_error')

    print(f"  Worst MAE range: {worst_df['mean_abs_error'].min():.2f}m - {worst_df['mean_abs_error'].max():.2f}m")

    frames = []
    for i, (idx, row) in enumerate(tqdm(worst_df.iterrows(), total=len(worst_df), desc="Generating worst frames")):
        frame_path = frames_dir / f"worst_frame_{i:04d}.png"

        success = plot_transect_visualization(row, frame_path, data_root)

        if success:
            frames.append(frame_path)

    if not frames:
        print("  Warning: No frames generated")
        return None

    print(f"  Generated {len(frames)} frames")
    print(f"  Creating GIF...")

    # Create GIF
    images = [Image.open(f) for f in frames]
    gif_path = output_dir / "worst_transects.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1500,  # 1.5 seconds per frame (slower to analyze)
        loop=0
    )

    print(f"✓ Saved worst transects GIF to {gif_path}")
    print(f"  Size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save worst transects info to CSV
    worst_info = worst_df[['AOI', 'TransectID', 'base_abs_error', 'top_abs_error', 'mean_abs_error']].copy()
    worst_info.to_csv(output_dir / 'worst_transects_info.csv', index=False)
    print(f"✓ Saved worst transects info to {output_dir / 'worst_transects_info.csv'}")

    # Clean up frames
    print("  Cleaning up frames...")
    for frame in frames:
        frame.unlink()
    frames_dir.rmdir()

    return gif_path


def main():
    print("="*80)
    print("Creating Output Figures and Visualizations")
    print("="*80)

    # Paths
    predictions_dir = Path("../outputs/predictions_validation")
    output_dir = Path("../outputs/visualizations_new_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all comparison data
    print("\nLoading comparison data...")
    df = load_all_comparisons(predictions_dir)

    if df is None:
        print("Error: No comparison data found")
        return

    print(f"  Loaded {len(df)} transects from {df['AOI'].nunique()} AOIs")
    print(f"  Total valid predictions: {df['base_abs_error'].notna().sum()}")

    # Create metrics summary
    print("\nCreating metrics summary...")
    metrics_df = create_metrics_summary(df, output_dir)

    # Print overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Base MAE:  {df['base_abs_error'].mean():.2f} m")
    print(f"Top MAE:   {df['top_abs_error'].mean():.2f} m")
    print(f"Base RMSE: {np.sqrt((df['base_error']**2).mean()):.2f} m")
    print(f"Top RMSE:  {np.sqrt((df['top_error']**2).mean()):.2f} m")
    print("="*80)

    # Create scatter plots
    print("\nCreating scatter plots...")
    create_scatter_plots(df, output_dir)

    # Create transect GIFs
    print("\n" + "="*80)
    print("Creating GIFs")
    print("="*80)

    create_transect_gif(df, output_dir, max_transects=100, step=1)
    create_worst_transects_gif(df, output_dir, n_worst=20)

    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - metrics_summary.png: Metrics plots and error distributions")
    print("  - scatter_plots.png: Predicted vs ground truth scatter plots")
    print("  - all_transects.gif: Animation of transect predictions")
    print("  - worst_transects.gif: Animation of worst performing transects")
    print("  - worst_transects_info.csv: Details of worst performing transects")
    print("="*80)


if __name__ == "__main__":
    main()
