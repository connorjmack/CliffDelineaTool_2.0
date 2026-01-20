#!/usr/bin/env python3
"""
Run CliffDelineaTool v1.0 on validation datasets (AOI 5-8)
and compare with v2.0 results
"""

import os
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from pathlib import Path

# v1.0 Parameters (default from paper)
PARAMS = {
    'nVert': 20,
    'baseMaxElev': 5,
    'baseSea': 15,
    'baseLand': 25,
    'topSea': 20,
    'topLand': 15,
    'propConvex': 0.5,
    'smoothWindow': 10
}

def process_transect_v1(sub, params):
    """
    Process a single transect using v1.0 logic
    Returns base and top point IDs
    """
    nVert = params['nVert']
    baseMaxElev = params['baseMaxElev']
    baseSea = params['baseSea']
    baseLand = params['baseLand']
    topSea = params['topSea']
    topLand = params['topLand']
    propConvex = params['propConvex']

    rowCount = len(sub)

    # Reset index first to ensure 0-based indexing
    sub = sub.reset_index(drop=True)

    # Fill data gaps
    sub.loc[sub['Elevation'] < -50, 'Elevation'] = np.nan
    sub['Elevation'] = sub['Elevation'].interpolate()
    sub['Elevation'] = sub['Elevation'].ffill().bfill()

    # Initialize slope columns
    sub['SeaSlope'] = 0.0
    sub['LandSlope'] = 0.0
    sub['Trendline1'] = 0.0
    sub['Difference1'] = 0.0

    # Calculate local slopes
    for z in range(nVert, rowCount - nVert):
        count = 0
        for s in range(1, nVert + 1):
            if sub.iloc[z]['Elevation'] != sub.iloc[z-s]['Elevation']:
                angle = math.degrees(math.atan(
                    (sub.iloc[z]['Elevation'] - sub.iloc[z-s]['Elevation']) /
                    (sub.iloc[z]['Distance'] - sub.iloc[z-s]['Distance'])
                ))
                if angle < 0:
                    angle = 0
                sub.loc[sub.index[z], 'SeaSlope'] += angle
                count += 1

        if count > 0:
            sub.loc[sub.index[z], 'SeaSlope'] /= count

        count = 0
        for s in range(1, nVert + 1):
            if sub.iloc[z]['Elevation'] != sub.iloc[z+s]['Elevation']:
                angle = math.degrees(math.atan(
                    (sub.iloc[z+s]['Elevation'] - sub.iloc[z]['Elevation']) /
                    (sub.iloc[z+s]['Distance'] - sub.iloc[z]['Distance'])
                ))
                if angle < 0:
                    angle = 0
                sub.loc[sub.index[z], 'LandSlope'] += angle
                count += 1

        if count > 0:
            sub.loc[sub.index[z], 'LandSlope'] /= count

    # Limit transect to highest point + nVert
    indMaxPos = sub['Elevation'].values.argmax()
    if rowCount > indMaxPos + nVert:
        allDrop = rowCount - (indMaxPos + nVert + 1)
        if allDrop > 0:
            sub = sub.iloc[:-allDrop]
            rowCount = len(sub)
            sub = sub.reset_index(drop=True)

    # Check if we have enough points
    if rowCount < 2:
        return None, None

    # Draw trendline #1 using iloc for safety
    sub.loc[sub.index[0], 'Trendline1'] = sub.iloc[0]['Elevation']
    sub.loc[sub.index[-1], 'Trendline1'] = sub.iloc[-1]['Elevation']

    for i in range(1, rowCount-1):
        sub.loc[sub.index[i], 'Trendline1'] = (
            (sub.iloc[i]['Distance'] - sub.iloc[0]['Distance']) *
            (sub.iloc[-1]['Elevation'] - sub.iloc[0]['Elevation']) /
            (sub.iloc[-1]['Distance'] - sub.iloc[0]['Distance']) +
            sub.iloc[0]['Elevation']
        )

    sub['Difference1'] = sub['Elevation'] - sub['Trendline1']

    # Find cliff base
    base_candidates = sub[
        (sub['Elevation'] < baseMaxElev) &
        (sub['SeaSlope'] < baseSea) &
        (sub['LandSlope'] > baseLand) &
        (sub['Difference1'] < 0)
    ]

    if len(base_candidates) > 0:
        base_idx = base_candidates['Difference1'].idxmin()
        base_point_id = sub.loc[base_idx, 'PointID']
        base_distance = sub.loc[base_idx, 'Distance']
    else:
        return None, None

    # Find cliff top
    top_candidates = sub[
        (sub['Distance'] > base_distance) &
        (sub['SeaSlope'] > topSea) &
        (sub['LandSlope'] < topLand) &
        (sub['Difference1'] > 0)
    ]

    if len(top_candidates) > 0:
        top_idx = top_candidates['Difference1'].idxmax()
        top_point_id = sub.loc[top_idx, 'PointID']
    else:
        return base_point_id, None

    return base_point_id, top_point_id


def run_v1_on_aoi(aoi_name, data_dir, output_dir):
    """Run v1.0 on a single AOI"""
    print(f"\n{'='*60}")
    print(f"Processing {aoi_name} with v1.0")
    print(f"{'='*60}")

    # Load data
    points_file = data_dir / aoi_name / f"{aoi_name}_points.txt"
    print(f"Loading: {points_file}")

    data = pd.read_csv(points_file)
    data.columns = ['PointID', 'TransectID', 'Elevation', 'Distance']

    print(f"  Total points: {len(data)}")
    print(f"  Transects: {data['TransectID'].nunique()}")

    # Process each transect
    results = []

    for transect_id in sorted(data['TransectID'].unique()):
        sub = data[data['TransectID'] == transect_id].copy()
        sub = sub.sort_values('Distance').reset_index(drop=True)

        if len(sub) > 0:
            base_id, top_id = process_transect_v1(sub, PARAMS)

            if base_id is not None:
                base_dist = sub[sub['PointID'] == base_id]['Distance'].values[0]
            else:
                base_dist = np.nan

            if top_id is not None:
                top_dist = sub[sub['PointID'] == top_id]['Distance'].values[0]
            else:
                top_dist = np.nan

            results.append({
                'TransectID': transect_id,
                'base_distance': base_dist,
                'top_distance': top_dist,
                'base_point_id': base_id,
                'top_point_id': top_id
            })

    results_df = pd.DataFrame(results)

    # Save results
    output_file = output_dir / f"{aoi_name}_v1_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"  Saved results to: {output_file}")
    print(f"  Valid predictions: {results_df['base_distance'].notna().sum()} bases, {results_df['top_distance'].notna().sum()} tops")

    return results_df


def main():
    """Main execution"""
    # Setup paths
    base_dir = Path("/Users/cjmack/Documents/GitHub/CliffDelineaTool_2.0")
    data_dir = base_dir / "datasets" / "validation"
    output_dir = base_dir / "v1_outputs"
    output_dir.mkdir(exist_ok=True)

    # Process each validation AOI
    aois = ['aoi5', 'aoi6', 'aoi7', 'aoi8']

    all_results = {}

    for aoi in aois:
        results = run_v1_on_aoi(aoi, data_dir, output_dir)
        all_results[aoi] = results

    print(f"\n{'='*60}")
    print("v1.0 Processing Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")

    # Summary statistics
    print("\nSummary:")
    for aoi, results in all_results.items():
        n_transects = len(results)
        n_valid_base = results['base_distance'].notna().sum()
        n_valid_top = results['top_distance'].notna().sum()
        print(f"  {aoi}: {n_transects} transects, {n_valid_base} bases, {n_valid_top} tops")


if __name__ == "__main__":
    main()
