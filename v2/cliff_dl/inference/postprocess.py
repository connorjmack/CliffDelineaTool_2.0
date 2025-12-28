"""
Post-processing for cliff detection predictions.

Includes:
- Alongshore outlier removal (from v1.0)
- Smoothing
- Constraint enforcement
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def remove_alongshore_outliers(
    predictions_df: pd.DataFrame,
    smooth_window: int = 10,
    outlier_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Remove alongshore outliers using v1.0 methodology.

    Uses moving median smoothing and OLS regression with standardized
    residuals to identify and replace outliers.

    Args:
        predictions_df: DataFrame with columns:
            - TransectID
            - base_distance
            - top_distance
        smooth_window: Moving median window size
        outlier_threshold: Standardized residual threshold

    Returns:
        Cleaned predictions DataFrame
    """
    df = predictions_df.copy()

    # Sort by TransectID (alongshore order)
    df = df.sort_values('TransectID').reset_index(drop=True)

    # Filter valid predictions only
    valid_base = df['base_distance'] > 0
    valid_top = df['top_distance'] > 0

    if valid_base.sum() < smooth_window or valid_top.sum() < smooth_window:
        print(f"Warning: Not enough valid predictions for smoothing "
              f"(base: {valid_base.sum()}, top: {valid_top.sum()})")
        return df

    # Process base distances
    if valid_base.any():
        df = _smooth_and_remove_outliers(
            df,
            distance_col='base_distance',
            valid_mask=valid_base,
            smooth_window=smooth_window,
            outlier_threshold=outlier_threshold
        )

    # Process top distances
    if valid_top.any():
        df = _smooth_and_remove_outliers(
            df,
            distance_col='top_distance',
            valid_mask=valid_top,
            smooth_window=smooth_window,
            outlier_threshold=outlier_threshold
        )

    return df


def _smooth_and_remove_outliers(
    df: pd.DataFrame,
    distance_col: str,
    valid_mask: pd.Series,
    smooth_window: int,
    outlier_threshold: float
) -> pd.DataFrame:
    """
    Helper function to smooth and remove outliers for a single distance column.
    """
    # Compute smoothed distances using moving median
    df[f'{distance_col}_smoothed'] = df[distance_col].copy()

    # Apply rolling median only to valid values
    valid_distances = df.loc[valid_mask, distance_col]

    if len(valid_distances) >= smooth_window:
        smoothed = valid_distances.rolling(
            window=smooth_window,
            center=True,
            min_periods=1
        ).median()

        df.loc[valid_mask, f'{distance_col}_smoothed'] = smoothed

    # Compute standardized residuals using OLS
    try:
        # Prepare data for OLS
        y = df.loc[valid_mask, distance_col].values
        X = df.loc[valid_mask, f'{distance_col}_smoothed'].values

        # Fit OLS model
        model = sm.OLS(y, X).fit()
        influence = model.get_influence()

        # Get standardized residuals
        residuals = influence.resid_studentized_internal

        # Identify outliers
        outlier_mask = np.abs(residuals) > outlier_threshold

        # Get indices in original dataframe
        valid_indices = df.index[valid_mask]
        outlier_indices = valid_indices[outlier_mask]

        # Replace outliers with smoothed values
        df.loc[outlier_indices, distance_col] = df.loc[outlier_indices, f'{distance_col}_smoothed']

        print(f"  {distance_col}: Replaced {outlier_mask.sum()} outliers "
              f"out of {len(y)} valid predictions")

    except Exception as e:
        print(f"  Warning: Could not compute outliers for {distance_col}: {e}")

    return df


def apply_gaussian_process_smoothing(
    predictions_df: pd.DataFrame,
    length_scale: float = 5.0
) -> pd.DataFrame:
    """
    Apply Gaussian Process smoothing for ultra-smooth cliff lines.

    Alternative to moving median. More computationally expensive but
    produces smoother results.

    Args:
        predictions_df: DataFrame with TransectID, base_distance, top_distance
        length_scale: GP kernel length scale (controls smoothness)

    Returns:
        Smoothed predictions DataFrame
    """
    df = predictions_df.copy()

    # Sort by TransectID
    df = df.sort_values('TransectID').reset_index(drop=True)

    # Smooth base distances
    valid_base = df['base_distance'] > 0
    if valid_base.sum() > 3:
        df.loc[valid_base, 'base_distance'] = _gp_smooth(
            df.loc[valid_base, 'TransectID'].values,
            df.loc[valid_base, 'base_distance'].values,
            length_scale
        )

    # Smooth top distances
    valid_top = df['top_distance'] > 0
    if valid_top.sum() > 3:
        df.loc[valid_top, 'top_distance'] = _gp_smooth(
            df.loc[valid_top, 'TransectID'].values,
            df.loc[valid_top, 'top_distance'].values,
            length_scale
        )

    return df


def _gp_smooth(
    transect_ids: np.ndarray,
    distances: np.ndarray,
    length_scale: float
) -> np.ndarray:
    """
    Helper function for GP smoothing.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    kernel = RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1.0, normalize_y=True)

    X = transect_ids.reshape(-1, 1)
    y = distances

    gp.fit(X, y)
    smoothed = gp.predict(X)

    return smoothed


def apply_geomorphological_constraints(
    predictions_df: pd.DataFrame,
    transect_data: dict,
    max_base_elevation: float = 15.0,
    max_cliff_width: float = 150.0,
    typical_cliff_width: float = 100.0
) -> pd.DataFrame:
    """
    Apply geomorphological constraints based on coastal physics.

    PHASE 1 CONSTRAINTS:
    1. Cliff bases occur at low elevations (typically 0-15m)
    2. Cliff widths are reasonable (10-150m, typically ~100m)
    3. Base must be seaward of and lower elevation than top

    Args:
        predictions_df: DataFrame with TransectID, base_distance, top_distance
        transect_data: Dictionary mapping transect_id -> transect data dict with:
            - 'distances': array of distances from sea
            - 'features': array of features (first column is normalized elevation)
            - 'elevations': array of actual elevations (if available)
        max_base_elevation: Maximum elevation for cliff base (meters)
        max_cliff_width: Maximum cliff width (meters)
        typical_cliff_width: Typical cliff width for correction (meters)

    Returns:
        Corrected predictions DataFrame
    """
    df = predictions_df.copy()
    n_corrected_base = 0
    n_corrected_top = 0
    n_corrected_width = 0

    for idx, row in df.iterrows():
        transect_id = row['TransectID']
        base_pred = row['base_distance']
        top_pred = row['top_distance']

        # Skip if no valid predictions
        if base_pred <= 0 or top_pred <= 0:
            continue

        # Get transect data
        if transect_id not in transect_data:
            continue

        data = transect_data[transect_id]
        distances = data['distances']

        # Get elevation profile (denormalize if needed)
        if 'elevations' in data and data['elevations'] is not None:
            elevations = data['elevations']
        else:
            # Use normalized elevation scaled to approximate real elevations
            elev_norm = data['features'][:, 0]
            elevations = elev_norm * 50  # Approximate denormalization

        # CONSTRAINT 1: Check base elevation
        base_idx = np.argmin(np.abs(distances - base_pred))
        base_elev = elevations[base_idx]

        if base_elev > max_base_elevation:
            # Base is too high - find steep slope in low elevation zone
            low_elev_mask = elevations < max_base_elevation

            if low_elev_mask.any():
                # Compute slope magnitude
                gradients = np.abs(np.gradient(elevations, distances))

                # Find steepest point in low elevation zone
                low_elev_indices = np.where(low_elev_mask)[0]
                steep_idx = low_elev_indices[np.argmax(gradients[low_elev_indices])]

                # Update base prediction
                base_pred = distances[steep_idx]
                n_corrected_base += 1

        # CONSTRAINT 2: Check cliff width
        cliff_width = top_pred - base_pred

        if cliff_width > max_cliff_width:
            # Cliff too wide - cap at typical width
            top_pred = base_pred + typical_cliff_width
            n_corrected_width += 1
            n_corrected_top += 1
        elif cliff_width < 0:
            # Base ahead of top - fix by using typical width
            top_pred = base_pred + typical_cliff_width
            n_corrected_top += 1

        # CONSTRAINT 3: Ensure base elevation < top elevation
        top_idx = np.argmin(np.abs(distances - top_pred))
        top_elev = elevations[top_idx]

        # Recompute base_idx after potential correction
        base_idx = np.argmin(np.abs(distances - base_pred))
        base_elev = elevations[base_idx]

        if base_elev >= top_elev:
            # Find first point landward of base with higher elevation
            landward_mask = distances > base_pred
            landward_indices = np.where(landward_mask)[0]

            if len(landward_indices) > 0:
                higher_elev = elevations[landward_indices] > base_elev
                if higher_elev.any():
                    higher_idx = landward_indices[np.where(higher_elev)[0][0]]
                    top_pred = distances[higher_idx]
                    n_corrected_top += 1

        # Update predictions
        df.at[idx, 'base_distance'] = base_pred
        df.at[idx, 'top_distance'] = top_pred

    if n_corrected_base > 0 or n_corrected_top > 0 or n_corrected_width > 0:
        print(f"  Geomorphological corrections:")
        print(f"    - Bases corrected (high elevation): {n_corrected_base}")
        print(f"    - Tops corrected (elevation/width): {n_corrected_top}")
        print(f"    - Widths capped (>{max_cliff_width}m): {n_corrected_width}")

    return df


def enforce_constraints(predictions_df: pd.DataFrame, min_distance: float = 5.0) -> pd.DataFrame:
    """
    Enforce physical constraints on predictions.

    Constraints:
    1. Top must be landward of base (top_distance > base_distance)
    2. Minimum distance between base and top

    Args:
        predictions_df: DataFrame with base_distance, top_distance
        min_distance: Minimum distance between base and top (meters)

    Returns:
        Cleaned predictions DataFrame
    """
    df = predictions_df.copy()

    valid = (df['base_distance'] > 0) & (df['top_distance'] > 0)

    # Check constraint 1: top landward of base
    violated_order = valid & (df['top_distance'] <= df['base_distance'])

    if violated_order.any():
        print(f"Warning: {violated_order.sum()} transects have top seaward of base - removing")
        df.loc[violated_order, ['base_distance', 'top_distance']] = -1.0

    # Update valid mask
    valid = (df['base_distance'] > 0) & (df['top_distance'] > 0)

    # Check constraint 2: minimum distance
    distance_diff = df.loc[valid, 'top_distance'] - df.loc[valid, 'base_distance']
    violated_min = distance_diff < min_distance

    if violated_min.any():
        violated_indices = df.index[valid][violated_min]
        print(f"Warning: {violated_min.sum()} transects have base-top distance < {min_distance}m - removing")
        df.loc[violated_indices, ['base_distance', 'top_distance']] = -1.0

    return df


def postprocess_predictions(
    predictions_df: pd.DataFrame,
    smooth_window: int = 10,
    outlier_threshold: float = 2.0,
    min_base_top_distance: float = 5.0,
    use_gp_smoothing: bool = False,
    gp_length_scale: float = 5.0
) -> pd.DataFrame:
    """
    Complete post-processing pipeline.

    Args:
        predictions_df: DataFrame with columns:
            - TransectID
            - base_distance
            - top_distance
        smooth_window: Moving median window size
        outlier_threshold: Standardized residual threshold
        min_base_top_distance: Minimum distance between base and top
        use_gp_smoothing: Use Gaussian Process smoothing (slower but smoother)
        gp_length_scale: GP kernel length scale

    Returns:
        Cleaned and smoothed predictions DataFrame
    """
    df = predictions_df.copy()

    print("Post-processing predictions...")
    print(f"  Input: {len(df)} transects")

    valid_before = ((df['base_distance'] > 0) & (df['top_distance'] > 0)).sum()
    print(f"  Valid predictions before processing: {valid_before}")

    # Step 1: Enforce constraints
    df = enforce_constraints(df, min_distance=min_base_top_distance)

    # Step 2: Remove alongshore outliers
    df = remove_alongshore_outliers(
        df,
        smooth_window=smooth_window,
        outlier_threshold=outlier_threshold
    )

    # Step 3: Optional GP smoothing
    if use_gp_smoothing:
        print("  Applying Gaussian Process smoothing...")
        df = apply_gaussian_process_smoothing(df, length_scale=gp_length_scale)

    valid_after = ((df['base_distance'] > 0) & (df['top_distance'] > 0)).sum()
    print(f"  Valid predictions after processing: {valid_after}")
    print(f"  Removed: {valid_before - valid_after} transects")

    return df


def predictions_to_dataframe(
    transect_ids: list,
    base_distances: np.ndarray,
    top_distances: np.ndarray,
    base_confidences: Optional[np.ndarray] = None,
    top_confidences: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Convert model predictions to DataFrame format.

    Args:
        transect_ids: List of transect IDs
        base_distances: [N] predicted base distances
        top_distances: [N] predicted top distances
        base_confidences: [N] optional confidence scores
        top_confidences: [N] optional confidence scores

    Returns:
        DataFrame with predictions
    """
    df = pd.DataFrame({
        'TransectID': transect_ids,
        'base_distance': base_distances,
        'top_distance': top_distances
    })

    if base_confidences is not None:
        df['base_confidence'] = base_confidences

    if top_confidences is not None:
        df['top_confidence'] = top_confidences

    return df
