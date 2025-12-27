"""
Feature engineering for cliff detection from 1D elevation transects.

This module computes 12 features per point:
- 6 geometric features (elevation, distance, derivatives)
- 6 domain features from v1.0 (slopes, trendlines, curvature)
"""

import numpy as np
import pandas as pd
import math
from typing import Tuple, Optional


def compute_local_slope(
    elevations: np.ndarray,
    distances: np.ndarray,
    n_vert: int = 20,
    direction: str = 'seaward'
) -> np.ndarray:
    """
    Compute local slope for each point (matches v1.0 implementation).

    Args:
        elevations: Array of elevation values [seq_len]
        distances: Array of distance values [seq_len]
        n_vert: Number of adjacent points to use for slope calculation
        direction: 'seaward' or 'landward'

    Returns:
        Array of slope values in degrees [seq_len]
    """
    seq_len = len(elevations)
    slopes = np.zeros(seq_len)

    for i in range(seq_len):
        if direction == 'seaward':
            # Average slope to n_vert seaward points
            start_idx = max(0, i - n_vert)
            end_idx = i
        else:  # landward
            # Average slope to n_vert landward points
            start_idx = i
            end_idx = min(seq_len, i + n_vert)

        if start_idx == end_idx:
            slopes[i] = 0.0
            continue

        # Compute average slope over window
        elev_diff = elevations[end_idx - 1] - elevations[start_idx]
        dist_diff = distances[end_idx - 1] - distances[start_idx]

        if dist_diff > 0:
            angle = math.degrees(math.atan(elev_diff / dist_diff))
            slopes[i] = max(0.0, angle)  # Clip negative slopes to 0 (matches v1.0)
        else:
            slopes[i] = 0.0

    return slopes


def compute_features(
    transect_df: pd.DataFrame,
    n_vert: int = 20
) -> np.ndarray:
    """
    Compute 12-dimensional feature vector for each point in transect.

    Args:
        transect_df: DataFrame with columns ['Elevation', 'Distance']
                    Can also have 'PointID', 'TransectID' (ignored for features)
        n_vert: Window size for local slope calculation

    Returns:
        Feature array of shape [seq_len, 12]

    Features:
        0: elevation_raw - Raw elevation (m)
        1: elevation_normalized - Min-max normalized [0,1]
        2: distance_from_sea - NEAR_DIST (m)
        3: distance_normalized - Normalized [0,1]
        4: elevation_gradient - First derivative
        5: elevation_curvature - Second derivative
        6: seaward_slope - Average slope to n_vert seaward points (degrees)
        7: landward_slope - Average slope to n_vert landward points (degrees)
        8: trendline1_deviation - Elevation minus linear trendline
        9: local_slope_change - Difference between landward and seaward slopes
        10: convexity_index - Signed curvature
        11: relative_elevation - Standardized elevation
    """
    # Extract arrays
    elev = transect_df['Elevation'].values.astype(np.float32)
    dist = transect_df['Distance'].values.astype(np.float32)

    seq_len = len(elev)

    # 1. Geometric features
    # Feature 0: Raw elevation
    elev_raw = elev

    # Feature 1: Normalized elevation
    elev_min, elev_max = elev.min(), elev.max()
    elev_range = elev_max - elev_min
    if elev_range > 1e-8:
        elev_norm = (elev - elev_min) / elev_range
    else:
        elev_norm = np.zeros_like(elev)

    # Feature 2: Raw distance
    dist_raw = dist

    # Feature 3: Normalized distance
    dist_max = dist.max()
    if dist_max > 1e-8:
        dist_norm = dist / dist_max
    else:
        dist_norm = np.zeros_like(dist)

    # Feature 4: Elevation gradient (first derivative)
    gradient = np.gradient(elev, dist)

    # Feature 5: Elevation curvature (second derivative)
    curvature = np.gradient(gradient, dist)

    # 2. Domain features from v1.0
    # Feature 6: Seaward slope
    seaward_slope = compute_local_slope(elev, dist, n_vert, 'seaward')

    # Feature 7: Landward slope
    landward_slope = compute_local_slope(elev, dist, n_vert, 'landward')

    # Feature 8: Trendline 1 deviation
    # Linear trendline from seaward to landward end
    trendline1 = np.linspace(elev[0], elev[-1], seq_len)
    trendline_dev = elev - trendline1

    # Feature 9: Local slope change
    slope_change = landward_slope - seaward_slope

    # Feature 10: Convexity index (normalized curvature)
    # Signed curvature: positive=convex, negative=concave
    denominator = (1 + gradient**2)**(3/2)
    convexity = np.where(
        denominator > 1e-8,
        curvature / denominator,
        0.0
    )

    # Feature 11: Relative elevation (standardized)
    elev_mean = elev.mean()
    elev_std = elev.std()
    if elev_std > 1e-8:
        rel_elevation = (elev - elev_mean) / elev_std
    else:
        rel_elevation = np.zeros_like(elev)

    # Stack all features
    features = np.stack([
        elev_raw,           # 0
        elev_norm,          # 1
        dist_raw,           # 2
        dist_norm,          # 3
        gradient,           # 4
        curvature,          # 5
        seaward_slope,      # 6
        landward_slope,     # 7
        trendline_dev,      # 8
        slope_change,       # 9
        convexity,          # 10
        rel_elevation       # 11
    ], axis=1)

    return features.astype(np.float32)


def clean_transect_data(
    transect_df: pd.DataFrame,
    nodata_value: float = -9999.0,
    min_valid_elev: float = -50.0
) -> pd.DataFrame:
    """
    Clean transect data following v1.0 preprocessing.

    Args:
        transect_df: DataFrame with 'Elevation' column
        nodata_value: Value representing missing data
        min_valid_elev: Minimum valid elevation (values below are treated as missing)

    Returns:
        Cleaned DataFrame with filled elevations
    """
    df = transect_df.copy()

    # Mark invalid values as NaN
    df.loc[df['Elevation'] == nodata_value, 'Elevation'] = np.nan
    df.loc[df['Elevation'] < min_valid_elev, 'Elevation'] = np.nan

    # Fill missing data (matches v1.0 line 47-51)
    # 1. Linear interpolation
    df['Elevation'] = df['Elevation'].interpolate(method='linear')

    # 2. Forward fill for remaining NaNs
    df['Elevation'] = df['Elevation'].fillna(method='ffill')

    # 3. Backward fill for remaining NaNs
    df['Elevation'] = df['Elevation'].fillna(method='bfill')

    return df


def create_soft_labels(
    distances: np.ndarray,
    base_dist: float,
    top_dist: float,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Create soft (Gaussian-smoothed) segmentation labels.

    Instead of hard 0/1 labels at a single point, creates a Gaussian
    distribution centered on the ground truth position.

    Args:
        distances: Array of NEAR_DIST values [seq_len]
        base_dist: Ground truth cliff base distance
        top_dist: Ground truth cliff top distance
        sigma: Standard deviation of Gaussian (meters)

    Returns:
        Soft labels array of shape [seq_len, 3] with columns:
            [background, cliff_base, cliff_top]
    """
    seq_len = len(distances)

    # Gaussian around base
    base_labels = np.exp(-0.5 * ((distances - base_dist) / sigma) ** 2)

    # Gaussian around top
    top_labels = np.exp(-0.5 * ((distances - top_dist) / sigma) ** 2)

    # Background is complement
    max_cliff = np.maximum(base_labels, top_labels)
    background_labels = 1.0 - max_cliff

    # Stack into [seq_len, 3]
    soft_labels = np.stack([background_labels, base_labels, top_labels], axis=1)

    return soft_labels.astype(np.float32)


def create_hard_labels(
    distances: np.ndarray,
    base_dist: float,
    top_dist: float,
    tolerance: float = 2.0
) -> np.ndarray:
    """
    Create hard (binary) segmentation labels.

    Args:
        distances: Array of NEAR_DIST values [seq_len]
        base_dist: Ground truth cliff base distance
        top_dist: Ground truth cliff top distance
        tolerance: Distance window for positive labels (meters)

    Returns:
        Hard labels array of shape [seq_len] with values:
            0: background
            1: cliff_base
            2: cliff_top
    """
    seq_len = len(distances)
    labels = np.zeros(seq_len, dtype=np.int64)

    # Mark points within tolerance of base
    base_mask = np.abs(distances - base_dist) <= tolerance
    labels[base_mask] = 1

    # Mark points within tolerance of top (overrides base if overlapping)
    top_mask = np.abs(distances - top_dist) <= tolerance
    labels[top_mask] = 2

    return labels


def create_regression_targets(
    distances: np.ndarray,
    base_dist: float,
    top_dist: float
) -> np.ndarray:
    """
    Create regression targets (distance offsets to cliff positions).

    Args:
        distances: Array of NEAR_DIST values [seq_len]
        base_dist: Ground truth cliff base distance
        top_dist: Ground truth cliff top distance

    Returns:
        Regression targets array of shape [seq_len, 2] with columns:
            [base_offset, top_offset]
        where offset = ground_truth_distance - current_distance
    """
    base_offsets = base_dist - distances
    top_offsets = top_dist - distances

    regression_targets = np.stack([base_offsets, top_offsets], axis=1)

    return regression_targets.astype(np.float32)


def preprocess_transect(
    transect_df: pd.DataFrame,
    base_dist: Optional[float] = None,
    top_dist: Optional[float] = None,
    n_vert: int = 20,
    seg_tolerance: float = 2.0,
    use_soft_labels: bool = True
) -> dict:
    """
    Complete preprocessing pipeline for a single transect.

    Args:
        transect_df: DataFrame with columns ['PointID', 'TransectID', 'Elevation', 'Distance']
        base_dist: Ground truth cliff base distance (None for inference)
        top_dist: Ground truth cliff top distance (None for inference)
        n_vert: Window size for local slope calculation
        seg_tolerance: Tolerance for soft/hard label creation
        use_soft_labels: Use soft (Gaussian) labels instead of hard labels

    Returns:
        Dictionary with:
            'features': [seq_len, 12] feature array
            'seg_labels': [seq_len, 3] soft labels or [seq_len] hard labels
            'reg_labels': [seq_len, 2] regression targets (if ground truth provided)
            'distances': [seq_len] NEAR_DIST values
            'transect_id': Transect ID
            'base_dist_gt': Ground truth base distance (if provided)
            'top_dist_gt': Ground truth top distance (if provided)
    """
    # Sort by distance (seaward to landward)
    df = transect_df.sort_values('Distance').reset_index(drop=True)

    # Clean elevation data
    df = clean_transect_data(df)

    # Compute features
    features = compute_features(df, n_vert=n_vert)

    # Get transect metadata
    transect_id = df['TransectID'].iloc[0] if 'TransectID' in df.columns else -1
    distances = df['Distance'].values

    result = {
        'features': features,
        'distances': distances.astype(np.float32),
        'transect_id': int(transect_id),
    }

    # Create labels if ground truth is provided
    if base_dist is not None and top_dist is not None:
        if use_soft_labels:
            seg_labels = create_soft_labels(distances, base_dist, top_dist, sigma=seg_tolerance)
        else:
            seg_labels = create_hard_labels(distances, base_dist, top_dist, tolerance=seg_tolerance)

        reg_labels = create_regression_targets(distances, base_dist, top_dist)

        result['seg_labels'] = seg_labels
        result['reg_labels'] = reg_labels
        result['base_dist_gt'] = float(base_dist)
        result['top_dist_gt'] = float(top_dist)

    return result
