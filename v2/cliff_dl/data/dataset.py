"""
PyTorch Dataset for cliff detection from 1D elevation transects.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .preprocessing import preprocess_transect


class CliffTransectDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed cliff transect data.

    Supports two modes:
    1. From preprocessed .pt files (fast)
    2. From raw shapefiles (slower, for first-time loading)
    """

    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        use_soft_labels: bool = True
    ):
        """
        Initialize dataset from preprocessed .pt file.

        Args:
            data_path: Path to preprocessed .pt file
            transform: Optional augmentation function
            use_soft_labels: Use soft (Gaussian) labels vs hard labels
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.use_soft_labels = use_soft_labels

        # Load preprocessed data (weights_only=False is safe for our own preprocessed data)
        self.transects = torch.load(self.data_path, weights_only=False)

        print(f"Loaded {len(self.transects)} transects from {self.data_path}")

    def __len__(self) -> int:
        return len(self.transects)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single transect.

        Returns:
            Dictionary with:
                'features': [seq_len, 12] feature tensor
                'seg_labels': [seq_len, 3] soft labels or [seq_len] hard labels
                'reg_labels': [seq_len, 2] regression targets
                'distances': [seq_len] NEAR_DIST values
                'transect_id': int
                'base_dist_gt': float (ground truth base distance)
                'top_dist_gt': float (ground truth top distance)
        """
        transect = self.transects[idx]

        features = torch.FloatTensor(transect['features'])
        distances = torch.FloatTensor(transect['distances'])

        # Load labels (check if they exist, for inference mode)
        if 'seg_labels' in transect:
            if self.use_soft_labels and transect['seg_labels'].ndim == 2:
                # Soft labels [seq_len, 3]
                seg_labels = torch.FloatTensor(transect['seg_labels'])
            else:
                # Hard labels [seq_len]
                seg_labels = torch.LongTensor(transect['seg_labels'])

            reg_labels = torch.FloatTensor(transect['reg_labels'])
            base_dist_gt = transect['base_dist_gt']
            top_dist_gt = transect['top_dist_gt']
        else:
            # Inference mode: no labels
            seg_labels = torch.zeros(len(features), dtype=torch.long)
            reg_labels = torch.zeros(len(features), 2, dtype=torch.float32)
            base_dist_gt = -1.0
            top_dist_gt = -1.0

        # Apply optional augmentation
        if self.transform:
            features, seg_labels = self.transform(features, seg_labels)

        return {
            'features': features,
            'seg_labels': seg_labels,
            'reg_labels': reg_labels,
            'distances': distances,
            'transect_id': transect['transect_id'],
            'base_dist_gt': base_dist_gt,
            'top_dist_gt': top_dist_gt
        }


def load_aoi_data(
    aoi_path: Path,
    n_vert: int = 20,
    seg_tolerance: float = 2.0,
    use_soft_labels: bool = True
) -> List[Dict]:
    """
    Load and preprocess data from a single AOI.

    Args:
        aoi_path: Path to AOI directory (e.g., datasets/calibration/aoi1)
        n_vert: Window size for local slope calculation
        seg_tolerance: Tolerance for label creation (meters)
        use_soft_labels: Use soft (Gaussian) labels

    Returns:
        List of preprocessed transect dictionaries
    """
    aoi_name = aoi_path.name
    print(f"Processing {aoi_name}...")

    # Load point data
    points_file = aoi_path / f"{aoi_name}_points.txt"
    if not points_file.exists():
        raise FileNotFoundError(f"Points file not found: {points_file}")

    points = pd.read_csv(points_file)
    # Rename columns to standardize (case-insensitive matching)
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

    # Load ground truth
    base_true_file = aoi_path / f"{aoi_name}_base_true.shp"
    top_true_file = aoi_path / f"{aoi_name}_top_true.shp"

    if not base_true_file.exists() or not top_true_file.exists():
        raise FileNotFoundError(f"Ground truth files not found for {aoi_name}")

    base_true = gpd.read_file(base_true_file)
    top_true = gpd.read_file(top_true_file)

    # Standardize column names (case-insensitive matching)
    col_map_base = {}
    for col in base_true.columns:
        col_upper = col.upper()
        if col_upper in ['ID_1', 'OBJECTID'] and 'TransectID' not in base_true.columns:
            col_map_base[col] = 'TransectID'
        elif col_upper == 'NEAR_DIST' and 'Distance' not in base_true.columns:
            col_map_base[col] = 'Distance'

    col_map_top = {}
    for col in top_true.columns:
        col_upper = col.upper()
        if col_upper in ['ID_1', 'OBJECTID'] and 'TransectID' not in top_true.columns:
            col_map_top[col] = 'TransectID'
        elif col_upper == 'NEAR_DIST' and 'Distance' not in top_true.columns:
            col_map_top[col] = 'Distance'

    if col_map_base:
        base_true.rename(columns=col_map_base, inplace=True)
    if col_map_top:
        top_true.rename(columns=col_map_top, inplace=True)

    # Process each transect
    transects = []
    transect_ids = points['TransectID'].unique()

    for transect_id in transect_ids:
        # Get points for this transect
        transect_points = points[points['TransectID'] == transect_id].copy()

        # Get ground truth distances
        base_row = base_true[base_true['TransectID'] == transect_id]
        top_row = top_true[top_true['TransectID'] == transect_id]

        if len(base_row) == 0 or len(top_row) == 0:
            print(f"Warning: No ground truth for transect {transect_id}, skipping")
            continue

        base_dist = base_row.iloc[0]['Distance']
        top_dist = top_row.iloc[0]['Distance']

        # Preprocess transect
        preprocessed = preprocess_transect(
            transect_points,
            base_dist=base_dist,
            top_dist=top_dist,
            n_vert=n_vert,
            seg_tolerance=seg_tolerance,
            use_soft_labels=use_soft_labels
        )

        transects.append(preprocessed)

    print(f"Processed {len(transects)} transects from {aoi_name}")
    return transects


def create_dataset_from_aois(
    data_root: str,
    aoi_list: List[int],
    output_path: str,
    dataset_type: str = 'calibration',
    n_vert: int = 20,
    seg_tolerance: float = 2.0,
    use_soft_labels: bool = True
):
    """
    Create preprocessed dataset from multiple AOIs.

    Args:
        data_root: Root directory of datasets (e.g., '../datasets')
        aoi_list: List of AOI numbers (e.g., [1, 2, 3])
        output_path: Where to save preprocessed .pt file
        dataset_type: 'calibration' or 'validation' (ignored if auto-detecting)
        n_vert: Window size for local slope calculation
        seg_tolerance: Tolerance for label creation
        use_soft_labels: Use soft (Gaussian) labels
    """
    data_root = Path(data_root)
    all_transects = []

    for aoi_num in aoi_list:
        # Auto-detect folder: AOIs 1-4 are in calibration, AOIs 5-8 are in validation
        if aoi_num <= 4:
            folder = 'calibration'
        else:
            folder = 'validation'

        aoi_path = data_root / folder / f"aoi{aoi_num}"
        if not aoi_path.exists():
            print(f"Warning: AOI path not found: {aoi_path}, skipping")
            continue

        transects = load_aoi_data(
            aoi_path,
            n_vert=n_vert,
            seg_tolerance=seg_tolerance,
            use_soft_labels=use_soft_labels
        )
        all_transects.extend(transects)

    # Save preprocessed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_transects, output_path)

    print(f"\nSaved {len(all_transects)} transects to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


# Augmentation functions
def add_gaussian_noise(features: torch.Tensor, noise_std: float = 0.02) -> torch.Tensor:
    """
    Add Gaussian noise to normalized elevation features.

    Args:
        features: [seq_len, 10] feature tensor (all normalized)
        noise_std: Standard deviation of noise (fraction of normalized range)

    Returns:
        Augmented features
    """
    features = features.clone()
    # Add noise to normalized elevation (feature 0)
    features[:, 0] += torch.randn_like(features[:, 0]) * noise_std
    # Clip to valid range [0, 1]
    features[:, 0] = torch.clamp(features[:, 0], 0.0, 1.0)
    return features


def vertical_shift(features: torch.Tensor, shift_range: float = 0.05) -> torch.Tensor:
    """
    Apply random vertical shift to normalized elevation (simulate datum changes).

    Args:
        features: [seq_len, 10] feature tensor (all normalized)
        shift_range: Maximum shift as fraction of normalized range

    Returns:
        Augmented features
    """
    features = features.clone()
    shift = torch.rand(1).item() * 2 * shift_range - shift_range
    features[:, 0] += shift  # Shift normalized elevation
    # Clip to valid range [0, 1]
    features[:, 0] = torch.clamp(features[:, 0], 0.0, 1.0)
    return features


class AugmentTransect:
    """
    Callable augmentation class for training.
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        shift_range: float = 1.0,
        noise_prob: float = 0.5,
        shift_prob: float = 0.5
    ):
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.noise_prob = noise_prob
        self.shift_prob = shift_prob

    def __call__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """
        Apply augmentations with given probabilities.

        Args:
            features: [seq_len, 10] (all normalized)
            labels: [seq_len, 3] or [seq_len]

        Returns:
            (augmented_features, labels)
        """
        if torch.rand(1).item() < self.noise_prob:
            features = add_gaussian_noise(features, self.noise_std)

        if torch.rand(1).item() < self.shift_prob:
            features = vertical_shift(features, self.shift_range)

        return features, labels
