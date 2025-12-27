"""
DataLoader utilities for batching variable-length transect sequences.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import List, Dict


def collate_transects(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length transects.

    Pads sequences to the same length and creates masks for valid positions.

    Args:
        batch: List of dictionaries from CliffTransectDataset.__getitem__()

    Returns:
        Batched dictionary with:
            'features': [batch, max_seq_len, 12] padded features
            'seg_labels': [batch, max_seq_len, 3] or [batch, max_seq_len] padded labels
            'reg_labels': [batch, max_seq_len, 2] padded regression targets
            'distances': [batch, max_seq_len] padded distances
            'mask': [batch, max_seq_len] boolean mask (True = valid, False = padding)
            'lengths': [batch] original sequence lengths
            'transect_ids': List[int] transect IDs
            'base_dist_gt': [batch] ground truth base distances
            'top_dist_gt': [batch] ground truth top distances
    """
    # Extract components from batch
    features = [item['features'] for item in batch]
    seg_labels = [item['seg_labels'] for item in batch]
    reg_labels = [item['reg_labels'] for item in batch]
    distances = [item['distances'] for item in batch]

    # Get original lengths
    lengths = torch.LongTensor([len(f) for f in features])

    # Pad sequences
    # Use 0 for features and regression, -100 for classification (ignored by loss)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Handle soft vs hard labels
    if seg_labels[0].dim() == 2:
        # Soft labels [seq_len, 3]
        seg_labels_padded = pad_sequence(seg_labels, batch_first=True, padding_value=0.0)
    else:
        # Hard labels [seq_len]
        seg_labels_padded = pad_sequence(seg_labels, batch_first=True, padding_value=-100)

    reg_labels_padded = pad_sequence(reg_labels, batch_first=True, padding_value=0.0)
    distances_padded = pad_sequence(distances, batch_first=True, padding_value=0.0)

    # Create mask for valid (non-padded) positions
    max_len = features_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    # Extract metadata
    transect_ids = [item['transect_id'] for item in batch]
    base_dist_gt = torch.FloatTensor([item['base_dist_gt'] for item in batch])
    top_dist_gt = torch.FloatTensor([item['top_dist_gt'] for item in batch])

    return {
        'features': features_padded,        # [batch, max_seq_len, 12]
        'seg_labels': seg_labels_padded,    # [batch, max_seq_len, 3] or [batch, max_seq_len]
        'reg_labels': reg_labels_padded,    # [batch, max_seq_len, 2]
        'distances': distances_padded,      # [batch, max_seq_len]
        'mask': mask,                       # [batch, max_seq_len]
        'lengths': lengths,                 # [batch]
        'transect_ids': transect_ids,       # List[int]
        'base_dist_gt': base_dist_gt,       # [batch]
        'top_dist_gt': top_dist_gt          # [batch]
    }


def create_data_loader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader with custom collate function.

    Args:
        dataset: CliffTransectDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_transects
    )


def get_batch_statistics(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics for a batch (useful for debugging).

    Args:
        batch: Batched dictionary from collate_transects()

    Returns:
        Dictionary with batch statistics
    """
    lengths = batch['lengths']
    features = batch['features']
    mask = batch['mask']

    stats = {
        'batch_size': len(lengths),
        'min_length': lengths.min().item(),
        'max_length': lengths.max().item(),
        'mean_length': lengths.float().mean().item(),
        'total_valid_points': mask.sum().item(),
        'total_padded_points': (~mask).sum().item(),
        'padding_fraction': (~mask).sum().item() / mask.numel(),
    }

    # Feature statistics (only on valid points)
    valid_features = features[mask]  # [total_valid, 12]
    if len(valid_features) > 0:
        stats['feature_mean'] = valid_features.mean(dim=0).tolist()
        stats['feature_std'] = valid_features.std(dim=0).tolist()

    return stats
