#!/usr/bin/env python3
"""
Compare CliffDelineaTool v1.0 vs v2.0 performance on validation datasets
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 10

def load_ground_truth(aoi_name, data_dir):
    """Load ground truth data from shapefiles"""
    base_file = data_dir / aoi_name / f"{aoi_name}_base_true.shp"
    top_file = data_dir / aoi_name / f"{aoi_name}_top_true.shp"

    base_gdf = gpd.read_file(base_file)
    top_gdf = gpd.read_file(top_file)

    # Find transect ID column (could be OBJECTID or Id_1 or Field2)
    transect_col = None
    for col in ['OBJECTID', 'Id_1', 'Field2', 'Field1']:
        if col in base_gdf.columns:
            transect_col = col
            break

    if transect_col is None:
        raise ValueError(f"Could not find transect ID column in {base_file}")

    # Extract transect ID and distance
    ground_truth = pd.DataFrame({
        'TransectID': base_gdf[transect_col].values,
        'base_true': base_gdf['NEAR_DIST'].values,
        'top_true': top_gdf['NEAR_DIST'].values
    })

    return ground_truth


def load_v1_results(aoi_name, v1_dir):
    """Load v1.0 predictions"""
    v1_file = v1_dir / f"{aoi_name}_v1_predictions.csv"
    v1_df = pd.read_csv(v1_file)

    # Rename for consistency
    v1_df = v1_df.rename(columns={
        'base_distance': 'base_v1',
        'top_distance': 'top_v1'
    })

    return v1_df[['TransectID', 'base_v1', 'top_v1']]


def load_v2_results(aoi_name, v2_dir):
    """Load v2.0 predictions"""
    v2_file = v2_dir / aoi_name / "predictions.csv"
    v2_df = pd.read_csv(v2_file)

    # Use smoothed predictions if available, otherwise raw
    v2_df['base_v2'] = v2_df.get('base_distance_smoothed', v2_df['base_distance'])
    v2_df['top_v2'] = v2_df.get('top_distance_smoothed', v2_df['top_distance'])

    return v2_df[['TransectID', 'base_v2', 'top_v2']]


def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'n': 0
        }

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n': len(y_true_clean)
    }


def create_comparison_plots(results_all, output_dir):
    """Create comprehensive comparison figures"""

    # Figure 1: Scatter plots - Base predictions
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('CliffDelineaTool v1.0 vs v2.0: Base Position Predictions', fontsize=16, fontweight='bold')

    aois = ['aoi5', 'aoi6', 'aoi7', 'aoi8']

    for i, aoi in enumerate(aois):
        df = results_all[aoi]

        # v1.0 scatter
        ax = axes[0, i]
        mask = ~(np.isnan(df['base_true']) | np.isnan(df['base_v1']))
        ax.scatter(df.loc[mask, 'base_true'], df.loc[mask, 'base_v1'],
                   alpha=0.5, s=30, color='steelblue', label='v1.0')

        # Perfect prediction line
        min_val = min(df['base_true'].min(), df['base_v1'].min())
        max_val = max(df['base_true'].max(), df['base_v1'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        metrics_v1 = calculate_metrics(df['base_true'].values, df['base_v1'].values)
        ax.set_title(f'{aoi.upper()} - v1.0\nMAE: {metrics_v1["mae"]:.2f}m, R²: {metrics_v1["r2"]:.3f}')
        ax.set_xlabel('Ground Truth (m)')
        ax.set_ylabel('Predicted (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # v2.0 scatter
        ax = axes[1, i]
        mask = ~(np.isnan(df['base_true']) | np.isnan(df['base_v2']))
        ax.scatter(df.loc[mask, 'base_true'], df.loc[mask, 'base_v2'],
                   alpha=0.5, s=30, color='forestgreen', label='v2.0')

        min_val = min(df['base_true'].min(), df['base_v2'].min())
        max_val = max(df['base_true'].max(), df['base_v2'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        metrics_v2 = calculate_metrics(df['base_true'].values, df['base_v2'].values)
        ax.set_title(f'{aoi.upper()} - v2.0\nMAE: {metrics_v2["mae"]:.2f}m, R²: {metrics_v2["r2"]:.3f}')
        ax.set_xlabel('Ground Truth (m)')
        ax.set_ylabel('Predicted (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_base_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'comparison_base_scatter.png'}")
    plt.close()

    # Figure 2: Scatter plots - Top predictions
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('CliffDelineaTool v1.0 vs v2.0: Top Position Predictions', fontsize=16, fontweight='bold')

    for i, aoi in enumerate(aois):
        df = results_all[aoi]

        # v1.0 scatter
        ax = axes[0, i]
        mask = ~(np.isnan(df['top_true']) | np.isnan(df['top_v1']))
        ax.scatter(df.loc[mask, 'top_true'], df.loc[mask, 'top_v1'],
                   alpha=0.5, s=30, color='steelblue', label='v1.0')

        min_val = min(df['top_true'].min(), df['top_v1'].min())
        max_val = max(df['top_true'].max(), df['top_v1'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        metrics_v1 = calculate_metrics(df['top_true'].values, df['top_v1'].values)
        ax.set_title(f'{aoi.upper()} - v1.0\nMAE: {metrics_v1["mae"]:.2f}m, R²: {metrics_v1["r2"]:.3f}')
        ax.set_xlabel('Ground Truth (m)')
        ax.set_ylabel('Predicted (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # v2.0 scatter
        ax = axes[1, i]
        mask = ~(np.isnan(df['top_true']) | np.isnan(df['top_v2']))
        ax.scatter(df.loc[mask, 'top_true'], df.loc[mask, 'top_v2'],
                   alpha=0.5, s=30, color='forestgreen', label='v2.0')

        min_val = min(df['top_true'].min(), df['top_v2'].min())
        max_val = max(df['top_true'].max(), df['top_v2'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        metrics_v2 = calculate_metrics(df['top_true'].values, df['top_v2'].values)
        ax.set_title(f'{aoi.upper()} - v2.0\nMAE: {metrics_v2["mae"]:.2f}m, R²: {metrics_v2["r2"]:.3f}')
        ax.set_xlabel('Ground Truth (m)')
        ax.set_ylabel('Predicted (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_top_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'comparison_top_scatter.png'}")
    plt.close()

    # Figure 3: Error distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Distributions: v1.0 vs v2.0', fontsize=16, fontweight='bold')

    # Combine all AOIs
    all_base_errors_v1 = []
    all_base_errors_v2 = []
    all_top_errors_v1 = []
    all_top_errors_v2 = []

    for aoi in aois:
        df = results_all[aoi]
        all_base_errors_v1.extend((df['base_v1'] - df['base_true']).dropna().values)
        all_base_errors_v2.extend((df['base_v2'] - df['base_true']).dropna().values)
        all_top_errors_v1.extend((df['top_v1'] - df['top_true']).dropna().values)
        all_top_errors_v2.extend((df['top_v2'] - df['top_true']).dropna().values)

    # Base errors - histogram
    ax = axes[0, 0]
    ax.hist(all_base_errors_v1, bins=30, alpha=0.6, color='steelblue', label='v1.0', edgecolor='black')
    ax.hist(all_base_errors_v2, bins=30, alpha=0.6, color='forestgreen', label='v2.0', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Base Position Errors (All AOIs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top errors - histogram
    ax = axes[0, 1]
    ax.hist(all_top_errors_v1, bins=30, alpha=0.6, color='steelblue', label='v1.0', edgecolor='black')
    ax.hist(all_top_errors_v2, bins=30, alpha=0.6, color='forestgreen', label='v2.0', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Top Position Errors (All AOIs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Base errors - box plot by AOI
    ax = axes[1, 0]
    base_errors_data = []
    base_labels = []
    for aoi in aois:
        df = results_all[aoi]
        base_errors_data.append((df['base_v1'] - df['base_true']).dropna().values)
        base_labels.append(f'{aoi}\nv1.0')
        base_errors_data.append((df['base_v2'] - df['base_true']).dropna().values)
        base_labels.append(f'{aoi}\nv2.0')

    bp = ax.boxplot(base_errors_data, labels=base_labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        if i % 2 == 0:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('forestgreen')
            patch.set_alpha(0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Prediction Error (m)')
    ax.set_title('Base Position Errors by AOI')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Top errors - box plot by AOI
    ax = axes[1, 1]
    top_errors_data = []
    top_labels = []
    for aoi in aois:
        df = results_all[aoi]
        top_errors_data.append((df['top_v1'] - df['top_true']).dropna().values)
        top_labels.append(f'{aoi}\nv1.0')
        top_errors_data.append((df['top_v2'] - df['top_true']).dropna().values)
        top_labels.append(f'{aoi}\nv2.0')

    bp = ax.boxplot(top_errors_data, labels=top_labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        if i % 2 == 0:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('forestgreen')
            patch.set_alpha(0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Prediction Error (m)')
    ax.set_title('Top Position Errors by AOI')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_error_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'comparison_error_distributions.png'}")
    plt.close()

    # Figure 4: Performance metrics summary
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Performance Metrics: v1.0 vs v2.0 (All Validation AOIs)', fontsize=16, fontweight='bold')

    metrics_summary = {
        'base_v1_mae': [],
        'base_v2_mae': [],
        'base_v1_rmse': [],
        'base_v2_rmse': [],
        'base_v1_r2': [],
        'base_v2_r2': [],
        'top_v1_mae': [],
        'top_v2_mae': [],
        'top_v1_rmse': [],
        'top_v2_rmse': [],
        'top_v1_r2': [],
        'top_v2_r2': [],
    }

    for aoi in aois:
        df = results_all[aoi]

        base_v1_metrics = calculate_metrics(df['base_true'].values, df['base_v1'].values)
        base_v2_metrics = calculate_metrics(df['base_true'].values, df['base_v2'].values)
        top_v1_metrics = calculate_metrics(df['top_true'].values, df['top_v1'].values)
        top_v2_metrics = calculate_metrics(df['top_true'].values, df['top_v2'].values)

        metrics_summary['base_v1_mae'].append(base_v1_metrics['mae'])
        metrics_summary['base_v2_mae'].append(base_v2_metrics['mae'])
        metrics_summary['base_v1_rmse'].append(base_v1_metrics['rmse'])
        metrics_summary['base_v2_rmse'].append(base_v2_metrics['rmse'])
        metrics_summary['base_v1_r2'].append(base_v1_metrics['r2'])
        metrics_summary['base_v2_r2'].append(base_v2_metrics['r2'])

        metrics_summary['top_v1_mae'].append(top_v1_metrics['mae'])
        metrics_summary['top_v2_mae'].append(top_v2_metrics['mae'])
        metrics_summary['top_v1_rmse'].append(top_v1_metrics['rmse'])
        metrics_summary['top_v2_rmse'].append(top_v2_metrics['rmse'])
        metrics_summary['top_v1_r2'].append(top_v1_metrics['r2'])
        metrics_summary['top_v2_r2'].append(top_v2_metrics['r2'])

    x = np.arange(len(aois))
    width = 0.35

    # MAE
    ax = axes[0]
    ax.bar(x - width/2, metrics_summary['base_v1_mae'], width, label='Base v1.0', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, metrics_summary['base_v2_mae'], width, label='Base v2.0', color='forestgreen', alpha=0.8)
    ax.set_xlabel('AOI')
    ax.set_ylabel('MAE (m)')
    ax.set_title('Mean Absolute Error (Base)')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in aois])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # RMSE
    ax = axes[1]
    ax.bar(x - width/2, metrics_summary['top_v1_mae'], width, label='Top v1.0', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, metrics_summary['top_v2_mae'], width, label='Top v2.0', color='forestgreen', alpha=0.8)
    ax.set_xlabel('AOI')
    ax.set_ylabel('MAE (m)')
    ax.set_title('Mean Absolute Error (Top)')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in aois])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # R²
    ax = axes[2]
    ax.bar(x - width/2, metrics_summary['base_v1_r2'], width, label='Base v1.0', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, metrics_summary['base_v2_r2'], width, label='Base v2.0', color='forestgreen', alpha=0.8)
    ax.set_xlabel('AOI')
    ax.set_ylabel('R²')
    ax.set_title('R² Score (Base)')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in aois])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics_summary.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'comparison_metrics_summary.png'}")
    plt.close()


def main():
    """Main execution"""
    print("="*80)
    print("CliffDelineaTool v1.0 vs v2.0 Comparison")
    print("="*80)

    # Setup paths
    base_dir = Path("/Users/cjmack/Documents/GitHub/CliffDelineaTool_2.0")
    data_dir = base_dir / "datasets" / "validation"
    v1_dir = base_dir / "v1_outputs"
    v2_dir = base_dir / "v2" / "outputs" / "predictions_validation"
    output_dir = base_dir / "comparison_outputs"
    output_dir.mkdir(exist_ok=True)

    aois = ['aoi5', 'aoi6', 'aoi7', 'aoi8']

    # Load all data
    print("\n[1] Loading data...")
    results_all = {}
    metrics_table = []

    for aoi in aois:
        print(f"\n  Processing {aoi}...")

        # Load ground truth
        gt = load_ground_truth(aoi, data_dir)
        print(f"    Ground truth: {len(gt)} transects")

        # Load v1.0 results
        v1 = load_v1_results(aoi, v1_dir)
        print(f"    v1.0 results: {v1['base_v1'].notna().sum()} bases, {v1['top_v1'].notna().sum()} tops")

        # Load v2.0 results
        v2 = load_v2_results(aoi, v2_dir)
        print(f"    v2.0 results: {v2['base_v2'].notna().sum()} bases, {v2['top_v2'].notna().sum()} tops")

        # Merge all
        merged = gt.merge(v1, on='TransectID', how='outer')
        merged = merged.merge(v2, on='TransectID', how='outer')
        results_all[aoi] = merged

        # Calculate metrics
        base_v1_metrics = calculate_metrics(merged['base_true'].values, merged['base_v1'].values)
        base_v2_metrics = calculate_metrics(merged['base_true'].values, merged['base_v2'].values)
        top_v1_metrics = calculate_metrics(merged['top_true'].values, merged['top_v1'].values)
        top_v2_metrics = calculate_metrics(merged['top_true'].values, merged['top_v2'].values)

        metrics_table.append({
            'AOI': aoi.upper(),
            'Feature': 'Base',
            'v1.0 MAE': base_v1_metrics['mae'],
            'v2.0 MAE': base_v2_metrics['mae'],
            'v1.0 RMSE': base_v1_metrics['rmse'],
            'v2.0 RMSE': base_v2_metrics['rmse'],
            'v1.0 R²': base_v1_metrics['r2'],
            'v2.0 R²': base_v2_metrics['r2'],
        })

        metrics_table.append({
            'AOI': aoi.upper(),
            'Feature': 'Top',
            'v1.0 MAE': top_v1_metrics['mae'],
            'v2.0 MAE': top_v2_metrics['mae'],
            'v1.0 RMSE': top_v1_metrics['rmse'],
            'v2.0 RMSE': top_v2_metrics['rmse'],
            'v1.0 R²': top_v1_metrics['r2'],
            'v2.0 R²': top_v2_metrics['r2'],
        })

    # Create metrics table
    metrics_df = pd.DataFrame(metrics_table)
    metrics_df.to_csv(output_dir / 'metrics_comparison.csv', index=False, float_format='%.3f')
    print(f"\n[2] Metrics table saved to: {output_dir / 'metrics_comparison.csv'}")

    # Print metrics table
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(metrics_df.to_string(index=False))

    # Create comparison plots
    print("\n[3] Creating comparison figures...")
    create_comparison_plots(results_all, output_dir)

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS (All Validation AOIs Combined)")
    print("="*80)

    all_base_true_v1 = []
    all_base_pred_v1 = []
    all_base_true_v2 = []
    all_base_pred_v2 = []
    all_top_true_v1 = []
    all_top_pred_v1 = []
    all_top_true_v2 = []
    all_top_pred_v2 = []

    for aoi in aois:
        df = results_all[aoi]

        # Base v1.0 - only matched pairs
        mask_base_v1 = ~(np.isnan(df['base_true']) | np.isnan(df['base_v1']))
        all_base_true_v1.extend(df.loc[mask_base_v1, 'base_true'].values)
        all_base_pred_v1.extend(df.loc[mask_base_v1, 'base_v1'].values)

        # Base v2.0 - only matched pairs
        mask_base_v2 = ~(np.isnan(df['base_true']) | np.isnan(df['base_v2']))
        all_base_true_v2.extend(df.loc[mask_base_v2, 'base_true'].values)
        all_base_pred_v2.extend(df.loc[mask_base_v2, 'base_v2'].values)

        # Top v1.0 - only matched pairs
        mask_top_v1 = ~(np.isnan(df['top_true']) | np.isnan(df['top_v1']))
        all_top_true_v1.extend(df.loc[mask_top_v1, 'top_true'].values)
        all_top_pred_v1.extend(df.loc[mask_top_v1, 'top_v1'].values)

        # Top v2.0 - only matched pairs
        mask_top_v2 = ~(np.isnan(df['top_true']) | np.isnan(df['top_v2']))
        all_top_true_v2.extend(df.loc[mask_top_v2, 'top_true'].values)
        all_top_pred_v2.extend(df.loc[mask_top_v2, 'top_v2'].values)

    overall_base_v1 = calculate_metrics(np.array(all_base_true_v1), np.array(all_base_pred_v1))
    overall_base_v2 = calculate_metrics(np.array(all_base_true_v2), np.array(all_base_pred_v2))
    overall_top_v1 = calculate_metrics(np.array(all_top_true_v1), np.array(all_top_pred_v1))
    overall_top_v2 = calculate_metrics(np.array(all_top_true_v2), np.array(all_top_pred_v2))

    print("\nBASE POSITION:")
    print(f"  v1.0: MAE={overall_base_v1['mae']:.2f}m, RMSE={overall_base_v1['rmse']:.2f}m, R²={overall_base_v1['r2']:.3f} (n={overall_base_v1['n']})")
    print(f"  v2.0: MAE={overall_base_v2['mae']:.2f}m, RMSE={overall_base_v2['rmse']:.2f}m, R²={overall_base_v2['r2']:.3f} (n={overall_base_v2['n']})")
    improvement_base = ((overall_base_v1['mae'] - overall_base_v2['mae']) / overall_base_v1['mae']) * 100
    print(f"  Improvement: {improvement_base:+.1f}% MAE reduction")

    print("\nTOP POSITION:")
    print(f"  v1.0: MAE={overall_top_v1['mae']:.2f}m, RMSE={overall_top_v1['rmse']:.2f}m, R²={overall_top_v1['r2']:.3f} (n={overall_top_v1['n']})")
    print(f"  v2.0: MAE={overall_top_v2['mae']:.2f}m, RMSE={overall_top_v2['rmse']:.2f}m, R²={overall_top_v2['r2']:.3f} (n={overall_top_v2['n']})")
    improvement_top = ((overall_top_v1['mae'] - overall_top_v2['mae']) / overall_top_v1['mae']) * 100
    print(f"  Improvement: {improvement_top:+.1f}% MAE reduction")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - metrics_comparison.csv")
    print("  - comparison_base_scatter.png")
    print("  - comparison_top_scatter.png")
    print("  - comparison_error_distributions.png")
    print("  - comparison_metrics_summary.png")


if __name__ == "__main__":
    main()
