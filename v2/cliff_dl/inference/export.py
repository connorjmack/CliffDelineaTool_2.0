"""
Export predictions to GIS formats (shapefiles).
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional


def export_to_shapefiles(
    predictions_df: pd.DataFrame,
    points_shapefile: str,
    output_dir: str,
    base_filename: str = "base_predicted",
    top_filename: str = "top_predicted"
):
    """
    Export predictions to shapefiles compatible with v1.0 workflow.

    Matches predicted distances to original point geometries.

    Args:
        predictions_df: DataFrame with columns:
            - TransectID
            - base_distance
            - top_distance
        points_shapefile: Path to original aoiX_points.shp
        output_dir: Directory to save output shapefiles
        base_filename: Filename for base shapefile (without .shp)
        top_filename: Filename for top shapefile (without .shp)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting predictions to shapefiles...")
    print(f"  Points source: {points_shapefile}")
    print(f"  Output directory: {output_dir}")

    # Load original points with geometries
    points_gdf = gpd.read_file(points_shapefile)

    # Standardize column names
    if 'ID_1' in points_gdf.columns and 'TransectID' not in points_gdf.columns:
        points_gdf.rename(columns={'ID_1': 'TransectID'}, inplace=True)
    if 'NEAR_DIST' in points_gdf.columns and 'Distance' not in points_gdf.columns:
        points_gdf.rename(columns={'NEAR_DIST': 'Distance'}, inplace=True)
    if 'RASTERVALU' in points_gdf.columns and 'Elevation' not in points_gdf.columns:
        points_gdf.rename(columns={'RASTERVALU': 'Elevation'}, inplace=True)

    # Export cliff base
    base_pred_df = predictions_df[predictions_df['base_distance'] > 0].copy()
    if len(base_pred_df) > 0:
        base_gdf = _match_predictions_to_points(
            points_gdf,
            base_pred_df,
            distance_col='base_distance',
            confidence_col='base_confidence' if 'base_confidence' in base_pred_df.columns else None
        )

        output_path = output_dir / f"{base_filename}.shp"
        base_gdf.to_file(output_path)
        print(f"  Saved {len(base_gdf)} base points to {output_path}")
    else:
        print("  Warning: No valid base predictions to export")

    # Export cliff top
    top_pred_df = predictions_df[predictions_df['top_distance'] > 0].copy()
    if len(top_pred_df) > 0:
        top_gdf = _match_predictions_to_points(
            points_gdf,
            top_pred_df,
            distance_col='top_distance',
            confidence_col='top_confidence' if 'top_confidence' in top_pred_df.columns else None
        )

        output_path = output_dir / f"{top_filename}.shp"
        top_gdf.to_file(output_path)
        print(f"  Saved {len(top_gdf)} top points to {output_path}")
    else:
        print("  Warning: No valid top predictions to export")


def _match_predictions_to_points(
    points_gdf: gpd.GeoDataFrame,
    predictions_df: pd.DataFrame,
    distance_col: str,
    confidence_col: Optional[str] = None,
    distance_tolerance: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Match predicted distances to closest point geometries.

    Args:
        points_gdf: GeoDataFrame with all points
        predictions_df: DataFrame with TransectID and distance predictions
        distance_col: Column name for predicted distance
        confidence_col: Optional column name for confidence scores
        distance_tolerance: Maximum distance difference to match (meters)

    Returns:
        GeoDataFrame with matched geometries
    """
    matched_points = []

    for _, pred_row in predictions_df.iterrows():
        transect_id = pred_row['TransectID']
        pred_distance = pred_row[distance_col]

        # Get all points for this transect
        transect_points = points_gdf[points_gdf['TransectID'] == transect_id].copy()

        if len(transect_points) == 0:
            print(f"    Warning: No points found for transect {transect_id}")
            continue

        # Find closest point to predicted distance
        transect_points['dist_diff'] = (transect_points['Distance'] - pred_distance).abs()
        closest_idx = transect_points['dist_diff'].idxmin()
        closest_point = transect_points.loc[closest_idx]

        if closest_point['dist_diff'] > distance_tolerance:
            print(f"    Warning: No close match for transect {transect_id} "
                  f"(diff={closest_point['dist_diff']:.2f}m)")

        # Create output row
        matched_row = {
            'TransectID': transect_id,
            'Distance': pred_distance,
            'Elevation': closest_point['Elevation'] if 'Elevation' in closest_point else None,
            'geometry': closest_point['geometry']
        }

        if confidence_col and confidence_col in pred_row:
            matched_row['Confidence'] = pred_row[confidence_col]

        matched_points.append(matched_row)

    # Create GeoDataFrame
    if matched_points:
        matched_gdf = gpd.GeoDataFrame(
            matched_points,
            crs=points_gdf.crs,
            geometry='geometry'
        )
    else:
        # Empty GeoDataFrame with correct structure
        matched_gdf = gpd.GeoDataFrame(
            columns=['TransectID', 'Distance', 'Elevation', 'geometry'],
            crs=points_gdf.crs
        )

    return matched_gdf


def export_to_csv(
    predictions_df: pd.DataFrame,
    output_path: str
):
    """
    Export predictions to CSV file.

    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_path, index=False)
    print(f"Exported predictions to {output_path}")


def compare_with_ground_truth(
    predictions_df: pd.DataFrame,
    base_true_shapefile: str,
    top_true_shapefile: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare predictions with ground truth and compute errors.

    Args:
        predictions_df: DataFrame with predicted distances
        base_true_shapefile: Path to base ground truth shapefile
        top_true_shapefile: Path to top ground truth shapefile
        output_path: Optional path to save comparison CSV

    Returns:
        DataFrame with predictions, ground truth, and errors
    """
    # Load ground truth
    base_true = gpd.read_file(base_true_shapefile)
    top_true = gpd.read_file(top_true_shapefile)

    # Standardize column names
    if 'ID_1' in base_true.columns:
        base_true.rename(columns={'ID_1': 'TransectID'}, inplace=True)
        top_true.rename(columns={'ID_1': 'TransectID'}, inplace=True)
    if 'NEAR_DIST' in base_true.columns:
        base_true.rename(columns={'NEAR_DIST': 'Distance'}, inplace=True)
        top_true.rename(columns={'NEAR_DIST': 'Distance'}, inplace=True)

    # Merge predictions with ground truth
    comparison_df = predictions_df.copy()

    comparison_df = comparison_df.merge(
        base_true[['TransectID', 'Distance']],
        on='TransectID',
        how='left',
        suffixes=('', '_base_true')
    )

    comparison_df = comparison_df.merge(
        top_true[['TransectID', 'Distance']],
        on='TransectID',
        how='left',
        suffixes=('', '_top_true')
    )

    # Rename ground truth columns
    comparison_df.rename(columns={
        'Distance': 'base_distance_true',
        'Distance_top_true': 'top_distance_true'
    }, inplace=True)

    # Compute errors
    comparison_df['base_error'] = (
        comparison_df['base_distance'] - comparison_df['base_distance_true']
    )
    comparison_df['top_error'] = (
        comparison_df['top_distance'] - comparison_df['top_distance_true']
    )

    comparison_df['base_abs_error'] = comparison_df['base_error'].abs()
    comparison_df['top_abs_error'] = comparison_df['top_error'].abs()

    # Save if requested
    if output_path:
        export_to_csv(comparison_df, output_path)

    return comparison_df
