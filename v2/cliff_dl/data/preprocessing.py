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
    Compute 10-dimensional feature vector for each point in transect.
    All features are normalized to similar scales for better training stability.

    Args:
        transect_df: DataFrame with columns ['Elevation', 'Distance']
                    Can also have 'PointID', 'TransectID' (ignored for features)
        n_vert: Window size for local slope calculation

    Returns:
        Feature array of shape [seq_len, 10]

    Features:
        0: elevation_normalized - Min-max normalized [0,1]
        1: distance_normalized - Normalized [0,1]
        2: elevation_gradient - First derivative (normalized by distance range)
        3: elevation_curvature - Second derivative (normalized by distance range^2)
        4: seaward_slope - Average slope to n_vert seaward points (normalized)
        5: landward_slope - Average slope to n_vert landward points (normalized)
        6: trendline1_deviation - Elevation minus linear trendline (normalized)
        7: local_slope_change - Difference between landward and seaward slopes (normalized)
        8: convexity_index - Signed curvature
        9: relative_elevation - Standardized elevation
    """
    # Extract arrays
    elev = transect_df['Elevation'].values.astype(np.float32)
    dist = transect_df['Distance'].values.astype(np.float32)

    seq_len = len(elev)

    # 1. Geometric features (all normalized to [0,1] or standardized)
    # Feature 0: Normalized elevation
    elev_min, elev_max = elev.min(), elev.max()
    elev_range = elev_max - elev_min
    if elev_range > 1e-8:
        elev_norm = (elev - elev_min) / elev_range
    else:
        elev_norm = np.zeros_like(elev)

    # Feature 1: Normalized distance
    dist_max = dist.max()
    if dist_max > 1e-8:
        dist_norm = dist / dist_max
    else:
        dist_norm = np.zeros_like(dist)

    # Feature 2: Elevation gradient (first derivative, normalized)
    gradient = np.gradient(elev, dist)
    gradient_std = gradient.std()
    if gradient_std > 1e-8:
        gradient_norm = gradient / gradient_std
    else:
        gradient_norm = np.zeros_like(gradient)

    # Feature 3: Elevation curvature (second derivative, normalized)
    curvature = np.gradient(gradient, dist)
    curvature_std = curvature.std()
    if curvature_std > 1e-8:
        curvature_norm = curvature / curvature_std
    else:
        curvature_norm = np.zeros_like(curvature)

    # 2. Domain features from v1.0 (normalized)
    # Feature 4: Seaward slope (normalized to [0,1])
    seaward_slope = compute_local_slope(elev, dist, n_vert, 'seaward')
    seaward_slope_norm = seaward_slope / 90.0  # Normalize degrees to [0,1]

    # Feature 5: Landward slope (normalized to [0,1])
    landward_slope = compute_local_slope(elev, dist, n_vert, 'landward')
    landward_slope_norm = landward_slope / 90.0  # Normalize degrees to [0,1]

    # Feature 6: Trendline deviation (normalized by elevation range)
    trendline1 = np.linspace(elev[0], elev[-1], seq_len)
    trendline_dev = elev - trendline1
    if elev_range > 1e-8:
        trendline_dev_norm = trendline_dev / elev_range
    else:
        trendline_dev_norm = np.zeros_like(trendline_dev)

    # Feature 7: Local slope change (normalized)
    slope_change = landward_slope - seaward_slope
    slope_change_norm = slope_change / 90.0  # Normalize to [-1, 1]

    # Feature 8: Convexity index (normalized curvature)
    # Signed curvature: positive=convex, negative=concave
    denominator = (1 + gradient**2)**(3/2)
    convexity = np.where(
        denominator > 1e-8,
        curvature / denominator,
        0.0
    )

    # Feature 9: Relative elevation (already standardized)
    elev_mean = elev.mean()
    elev_std = elev.std()
    if elev_std > 1e-8:
        rel_elevation = (elev - elev_mean) / elev_std
    else:
        rel_elevation = np.zeros_like(elev)

    # PHASE 2: Geomorphological features
    # Feature 10: Low elevation zone indicator (binary)
    # Cliff bases typically occur at low elevations (0-15m)
    low_elevation_zone = (elev < 15.0).astype(np.float32)

    # Feature 11: Shore proximity (exponential decay from sea)
    # Increases weight near shore where cliff base is expected
    dist_range = dist.max() - dist.min()
    if dist_range > 1e-8:
        # Normalize distances to [0, 1]
        dist_01 = (dist - dist.min()) / dist_range
        # Exponential decay: high near sea (0), low inland (1)
        # Decay factor of 0.05 means 95% weight lost at ~60% inland
        shore_proximity = np.exp(-5.0 * dist_01).astype(np.float32)
    else:
        shore_proximity = np.ones_like(dist, dtype=np.float32)

    # Feature 12: Maximum local slope (in 5-point window)
    # Cliffs have steep local slopes
    window = 5
    half_window = window // 2
    max_local_slope = np.zeros_like(gradient)
    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        max_local_slope[i] = np.abs(gradient[start:end]).max()

    # Normalize max_local_slope
    max_slope_range = max_local_slope.max() - max_local_slope.min()
    if max_slope_range > 1e-8:
        max_local_slope_norm = (max_local_slope - max_local_slope.min()) / max_slope_range
    else:
        max_local_slope_norm = np.zeros_like(max_local_slope)

    # Stack all features (13 features total, all normalized)
    features = np.stack([
        elev_norm,              # 0: Normalized elevation
        dist_norm,              # 1: Normalized distance
        gradient_norm,          # 2: Normalized gradient
        curvature_norm,         # 3: Normalized curvature
        seaward_slope_norm,     # 4: Normalized seaward slope
        landward_slope_norm,    # 5: Normalized landward slope
        trendline_dev_norm,     # 6: Normalized trendline deviation
        slope_change_norm,      # 7: Normalized slope change
        convexity,              # 8: Convexity index
        rel_elevation,          # 9: Relative elevation
        low_elevation_zone,     # 10: Low elevation zone indicator (PHASE 2)
        shore_proximity,        # 11: Shore proximity weight (PHASE 2)
        max_local_slope_norm    # 12: Maximum local slope (PHASE 2)
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
    df['Elevation'] = df['Elevation'].ffill()

    # 3. Backward fill for remaining NaNs
    df['Elevation'] = df['Elevation'].bfill()

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
    distribution centered on the ground truth position. Sigma is made
    adaptive to transect length for consistent learning across sequences.

    Args:
        distances: Array of NEAR_DIST values [seq_len]
        base_dist: Ground truth cliff base distance
        top_dist: Ground truth cliff top distance
        sigma: Base standard deviation of Gaussian (meters)

    Returns:
        Soft labels array of shape [seq_len, 3] with columns:
            [background, cliff_base, cliff_top]
    """
    seq_len = len(distances)

    # Make sigma adaptive to transect length
    # For a transect with 200 points and sigma=2.0, we want consistent behavior
    # Scale sigma proportionally to the average point spacing
    dist_range = distances.max() - distances.min()
    avg_point_spacing = dist_range / max(seq_len - 1, 1)
    # Adaptive sigma: maintain ~2-3 points within 1 sigma
    adaptive_sigma = max(sigma, avg_point_spacing * 2.0)

    # Gaussian around base
    base_labels = np.exp(-0.5 * ((distances - base_dist) / adaptive_sigma) ** 2)

    # Gaussian around top
    top_labels = np.exp(-0.5 * ((distances - top_dist) / adaptive_sigma) ** 2)

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
            'features': [seq_len, 10] feature array (all normalized)
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
