#!/usr/bin/env python3
"""
Create animated GIFs comparing v1.0 vs v2.0 predictions for each AOI.
One GIF per AOI showing transect-by-transect comparisons.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image

def load_data_for_aoi(aoi_name, data_dir, v1_dir, v2_dir):
    """Load all data for one AOI"""
    print(f"  Loading {aoi_name} data...")

    # Load ground truth
    base_gdf = gpd.read_file(data_dir / aoi_name / f"{aoi_name}_base_true.shp")
    top_gdf = gpd.read_file(data_dir / aoi_name / f"{aoi_name}_top_true.shp")

    # Find transect ID column
    transect_col = None
    for col in ['OBJECTID', 'Id_1', 'Field2', 'Field1']:
        if col in base_gdf.columns:
            transect_col = col
            break

    ground_truth = pd.DataFrame({
        'TransectID': base_gdf[transect_col].values,
        'base_true': base_gdf['NEAR_DIST'].values,
        'top_true': top_gdf['NEAR_DIST'].values
    })

    # Load transect points
    points_file = data_dir / aoi_name / f"{aoi_name}_points.txt"
    points_df = pd.read_csv(points_file)

    # Standardize column names
    if 'FID' in points_df.columns and 'OBJECTID' in points_df.columns:
        points_df = points_df.rename(columns={
            'FID': 'PointID',
            'OBJECTID': 'TransectID',
            'RASTERVALU': 'Elevation',
            'NEAR_DIST': 'Distance'
        })
    else:
        points_df.columns = ['PointID', 'TransectID', 'Elevation', 'Distance']

    # Load v1.0 predictions
    v1_df = pd.read_csv(v1_dir / f"{aoi_name}_v1_predictions.csv")
    v1_df = v1_df.rename(columns={'base_distance': 'base_v1', 'top_distance': 'top_v1'})

    # Load v2.0 predictions
    v2_df = pd.read_csv(v2_dir / aoi_name / "predictions.csv")
    v2_df['base_v2'] = v2_df.get('base_distance_smoothed', v2_df['base_distance'])
    v2_df['top_v2'] = v2_df.get('top_distance_smoothed', v2_df['top_distance'])

    # Merge predictions with ground truth
    results = ground_truth.merge(v1_df[['TransectID', 'base_v1', 'top_v1']], on='TransectID', how='left')
    results = results.merge(v2_df[['TransectID', 'base_v2', 'top_v2']], on='TransectID', how='left')

    return points_df, results


def create_transect_frame(ax, points_df, results_df, transect_id):
    """Create a single frame showing one transect"""
    ax.clear()

    # Get transect data
    transect_points = points_df[points_df['TransectID'] == transect_id].copy()
    transect_points = transect_points.sort_values('Distance')

    # Clean elevation data
    transect_points.loc[transect_points['Elevation'] < -50, 'Elevation'] = np.nan
    transect_points['Elevation'] = transect_points['Elevation'].interpolate().ffill().bfill()

    # Get predictions for this transect
    pred = results_df[results_df['TransectID'] == transect_id].iloc[0]

    # Plot elevation profile
    ax.plot(transect_points['Distance'], transect_points['Elevation'],
            'k-', linewidth=1.5, alpha=0.7, label='Elevation profile', zorder=1)
    ax.fill_between(transect_points['Distance'], transect_points['Elevation'],
                     alpha=0.1, color='gray', zorder=0)

    # Find y-range for markers
    elev_min = transect_points['Elevation'].min()
    elev_max = transect_points['Elevation'].max()
    marker_size = 100

    # Plot ground truth (black stars)
    if not np.isnan(pred['base_true']):
        idx = transect_points['Distance'].sub(pred['base_true']).abs().idxmin()
        base_elev = transect_points.loc[idx, 'Elevation']
        ax.scatter(pred['base_true'], base_elev, s=marker_size, marker='*',
                  color='black', edgecolor='white', linewidth=1.5,
                  label='Ground Truth Base', zorder=5)

    if not np.isnan(pred['top_true']):
        idx = transect_points['Distance'].sub(pred['top_true']).abs().idxmin()
        top_elev = transect_points.loc[idx, 'Elevation']
        ax.scatter(pred['top_true'], top_elev, s=marker_size, marker='*',
                  color='black', edgecolor='white', linewidth=1.5,
                  label='Ground Truth Top', zorder=5)

    # Plot v1.0 predictions (blue circles)
    if not np.isnan(pred['base_v1']):
        idx = transect_points['Distance'].sub(pred['base_v1']).abs().idxmin()
        base_elev_v1 = transect_points.loc[idx, 'Elevation']
        ax.scatter(pred['base_v1'], base_elev_v1, s=marker_size, marker='o',
                  color='steelblue', edgecolor='white', linewidth=1.5,
                  label='v1.0 Base', zorder=4)

    if not np.isnan(pred['top_v1']):
        idx = transect_points['Distance'].sub(pred['top_v1']).abs().idxmin()
        top_elev_v1 = transect_points.loc[idx, 'Elevation']
        ax.scatter(pred['top_v1'], top_elev_v1, s=marker_size, marker='o',
                  color='steelblue', edgecolor='white', linewidth=1.5,
                  label='v1.0 Top', zorder=4)

    # Plot v2.0 predictions (green triangles)
    if not np.isnan(pred['base_v2']):
        idx = transect_points['Distance'].sub(pred['base_v2']).abs().idxmin()
        base_elev_v2 = transect_points.loc[idx, 'Elevation']
        ax.scatter(pred['base_v2'], base_elev_v2, s=marker_size, marker='^',
                  color='forestgreen', edgecolor='white', linewidth=1.5,
                  label='v2.0 Base', zorder=4)

    if not np.isnan(pred['top_v2']):
        idx = transect_points['Distance'].sub(pred['top_v2']).abs().idxmin()
        top_elev_v2 = transect_points.loc[idx, 'Elevation']
        ax.scatter(pred['top_v2'], top_elev_v2, s=marker_size, marker='^',
                  color='forestgreen', edgecolor='white', linewidth=1.5,
                  label='v2.0 Top', zorder=4)

    # Calculate errors
    base_error_v1 = abs(pred['base_v1'] - pred['base_true']) if not np.isnan(pred['base_v1']) else np.nan
    base_error_v2 = abs(pred['base_v2'] - pred['base_true']) if not np.isnan(pred['base_v2']) else np.nan
    top_error_v1 = abs(pred['top_v1'] - pred['top_true']) if not np.isnan(pred['top_v1']) else np.nan
    top_error_v2 = abs(pred['top_v2'] - pred['top_true']) if not np.isnan(pred['top_v2']) else np.nan

    # Title with transect number and errors
    title = f'Transect {transect_id}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add error text box
    error_text = 'Errors (meters):\n'
    if not np.isnan(base_error_v1):
        error_text += f'Base v1.0: {base_error_v1:.1f}m\n'
    else:
        error_text += 'Base v1.0: N/A\n'

    if not np.isnan(base_error_v2):
        error_text += f'Base v2.0: {base_error_v2:.1f}m\n'
    else:
        error_text += 'Base v2.0: N/A\n'

    if not np.isnan(top_error_v1):
        error_text += f'Top v1.0: {top_error_v1:.1f}m\n'
    else:
        error_text += 'Top v1.0: N/A\n'

    if not np.isnan(top_error_v2):
        error_text += f'Top v2.0: {top_error_v2:.1f}m'
    else:
        error_text += 'Top v2.0: N/A'

    # Position text box in upper right
    ax.text(0.98, 0.97, error_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Labels and grid
    ax.set_xlabel('Distance from Sea (m)', fontsize=11)
    ax.set_ylabel('Elevation (m)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Legend (only show unique entries)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
             loc='upper left', fontsize=8, framealpha=0.9)

    # Set consistent y-axis limits with padding
    y_margin = (elev_max - elev_min) * 0.15
    ax.set_ylim(elev_min - y_margin, elev_max + y_margin)


def create_aoi_gif(aoi_name, data_dir, v1_dir, v2_dir, output_dir,
                   frame_duration=500, max_transects=None):
    """
    Create animated GIF for one AOI showing all transects.

    Args:
        aoi_name: Name of AOI (e.g., 'aoi5')
        data_dir: Path to validation data
        v1_dir: Path to v1 predictions
        v2_dir: Path to v2 predictions
        output_dir: Where to save GIF
        frame_duration: Milliseconds per frame
        max_transects: Maximum number of transects to include (None = all)
    """
    print(f"\n{'='*60}")
    print(f"Creating GIF for {aoi_name.upper()}")
    print(f"{'='*60}")

    # Load data
    points_df, results_df = load_data_for_aoi(aoi_name, data_dir, v1_dir, v2_dir)

    # Get transect IDs
    transect_ids = sorted(results_df['TransectID'].unique())

    # Limit transects if requested
    if max_transects is not None:
        # Sample evenly across the AOI
        step = max(1, len(transect_ids) // max_transects)
        transect_ids = transect_ids[::step][:max_transects]

    print(f"  Creating {len(transect_ids)} frames...")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'{aoi_name.upper()}: v1.0 vs v2.0 Cliff Detection',
                 fontsize=16, fontweight='bold')

    # Generate frames
    frames = []
    for idx, transect_id in enumerate(transect_ids):
        # Create frame
        create_transect_frame(ax, points_df, results_df, transect_id)

        # Convert plot to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        buf.close()

        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"    Frame {idx + 1}/{len(transect_ids)}")

    plt.close(fig)

    # Save as GIF
    output_file = output_dir / f'{aoi_name}_comparison.gif'
    print(f"  Saving GIF to: {output_file}")

    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=False
    )

    print(f"  ✓ Saved: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Calculate summary statistics for this AOI
    base_mae_v1 = results_df[['base_true', 'base_v1']].dropna().apply(
        lambda x: abs(x['base_v1'] - x['base_true']), axis=1).mean()
    base_mae_v2 = results_df[['base_true', 'base_v2']].dropna().apply(
        lambda x: abs(x['base_v2'] - x['base_true']), axis=1).mean()
    top_mae_v1 = results_df[['top_true', 'top_v1']].dropna().apply(
        lambda x: abs(x['top_v1'] - x['top_true']), axis=1).mean()
    top_mae_v2 = results_df[['top_true', 'top_v2']].dropna().apply(
        lambda x: abs(x['top_v2'] - x['top_true']), axis=1).mean()

    print(f"\n  Summary for {aoi_name.upper()}:")
    print(f"    Base MAE - v1.0: {base_mae_v1:.2f}m, v2.0: {base_mae_v2:.2f}m")
    print(f"    Top MAE  - v1.0: {top_mae_v1:.2f}m, v2.0: {top_mae_v2:.2f}m")


def main():
    """Main execution"""
    print("="*80)
    print("Creating Animated GIFs: v1.0 vs v2.0 Comparison")
    print("="*80)

    # Setup paths
    base_dir = Path("/Users/cjmack/Documents/GitHub/CliffDelineaTool_2.0")
    data_dir = base_dir / "datasets" / "validation"
    v1_dir = base_dir / "v1_outputs"
    v2_dir = base_dir / "v2" / "outputs" / "predictions_validation"
    output_dir = base_dir / "comparison_outputs"

    # Configuration
    aois = ['aoi5', 'aoi6', 'aoi7', 'aoi8']
    frame_duration = 500  # milliseconds per frame (500ms = 2 fps)
    max_transects = None  # None = include all transects, or set a number like 50

    print(f"\nSettings:")
    print(f"  Frame duration: {frame_duration}ms ({1000//frame_duration} fps)")
    print(f"  Max transects per AOI: {max_transects if max_transects else 'All'}")
    print(f"  Output directory: {output_dir}")

    # Create GIFs for each AOI
    for aoi in aois:
        try:
            create_aoi_gif(
                aoi,
                data_dir,
                v1_dir,
                v2_dir,
                output_dir,
                frame_duration=frame_duration,
                max_transects=max_transects
            )
        except Exception as e:
            print(f"  ✗ Error creating GIF for {aoi}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("GIF Creation Complete!")
    print("="*80)
    print(f"\nAll GIFs saved to: {output_dir}")
    print("\nFiles created:")
    for aoi in aois:
        gif_file = output_dir / f"{aoi}_comparison.gif"
        if gif_file.exists():
            print(f"  ✓ {gif_file.name} ({gif_file.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
