"""
Evaluation metrics for cliff detection.
"""

import numpy as np
import torch
from typing import Dict, Optional
from sklearn.metrics import r2_score


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute evaluation metrics following v1.0 methodology.

    Args:
        predictions: [N] predicted distances
        ground_truth: [N] ground truth distances
        prefix: Prefix for metric names (e.g., "base_" or "top_")

    Returns:
        Dictionary with metrics:
            - MAE: Mean Absolute Error (meters)
            - RMSE: Root Mean Squared Error (meters)
            - R2: R-squared score
            - Accuracy_1m: Percentage within 1 meter
            - Accuracy_2m: Percentage within 2 meters
            - Accuracy_5m: Percentage within 5 meters
    """
    # Filter out invalid predictions (-1)
    valid_mask = (predictions > 0) & (ground_truth > 0)

    if valid_mask.sum() == 0:
        return {
            f"{prefix}MAE": float('nan'),
            f"{prefix}RMSE": float('nan'),
            f"{prefix}R2": float('nan'),
            f"{prefix}Accuracy_1m": 0.0,
            f"{prefix}Accuracy_2m": 0.0,
            f"{prefix}Accuracy_5m": 0.0,
            f"{prefix}count": 0
        }

    pred_valid = predictions[valid_mask]
    gt_valid = ground_truth[valid_mask]

    # Compute errors
    errors = pred_valid - gt_valid
    abs_errors = np.abs(errors)

    # MAE
    mae = abs_errors.mean()

    # RMSE
    rmse = np.sqrt((errors ** 2).mean())

    # R² score
    try:
        r2 = r2_score(gt_valid, pred_valid)
    except:
        r2 = float('nan')

    # Accuracy within thresholds
    within_1m = (abs_errors <= 1.0).mean() * 100
    within_2m = (abs_errors <= 2.0).mean() * 100
    within_5m = (abs_errors <= 5.0).mean() * 100

    return {
        f"{prefix}MAE": float(mae),
        f"{prefix}RMSE": float(rmse),
        f"{prefix}R2": float(r2),
        f"{prefix}Accuracy_1m": float(within_1m),
        f"{prefix}Accuracy_2m": float(within_2m),
        f"{prefix}Accuracy_5m": float(within_5m),
        f"{prefix}count": int(valid_mask.sum())
    }


def compute_cliff_detection_metrics(
    base_predictions: np.ndarray,
    base_ground_truth: np.ndarray,
    top_predictions: np.ndarray,
    top_ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for both cliff base and top.
    Includes detection metrics (precision, recall, F1) in addition to position accuracy.

    Args:
        base_predictions: [N] predicted base distances
        base_ground_truth: [N] ground truth base distances
        top_predictions: [N] predicted top distances
        top_ground_truth: [N] ground truth top distances

    Returns:
        Dictionary with all metrics
    """
    # Compute metrics for base
    base_metrics = compute_metrics(base_predictions, base_ground_truth, prefix="base_")

    # Compute metrics for top
    top_metrics = compute_metrics(top_predictions, top_ground_truth, prefix="top_")

    # Combine
    all_metrics = {**base_metrics, **top_metrics}

    # Detection metrics (Precision, Recall, F1)
    # For base
    base_gt_has_cliff = (base_ground_truth > 0)
    base_pred_has_cliff = (base_predictions > 0)

    base_tp = (base_gt_has_cliff & base_pred_has_cliff).sum()
    base_fp = (~base_gt_has_cliff & base_pred_has_cliff).sum()
    base_fn = (base_gt_has_cliff & ~base_pred_has_cliff).sum()

    base_precision = base_tp / (base_tp + base_fp + 1e-8) * 100
    base_recall = base_tp / (base_tp + base_fn + 1e-8) * 100
    base_f1 = 2 * base_precision * base_recall / (base_precision + base_recall + 1e-8)

    all_metrics['base_precision'] = float(base_precision)
    all_metrics['base_recall'] = float(base_recall)
    all_metrics['base_f1'] = float(base_f1)
    all_metrics['base_detection_rate'] = float(base_recall)  # Detection rate = recall

    # For top
    top_gt_has_cliff = (top_ground_truth > 0)
    top_pred_has_cliff = (top_predictions > 0)

    top_tp = (top_gt_has_cliff & top_pred_has_cliff).sum()
    top_fp = (~top_gt_has_cliff & top_pred_has_cliff).sum()
    top_fn = (top_gt_has_cliff & ~top_pred_has_cliff).sum()

    top_precision = top_tp / (top_tp + top_fp + 1e-8) * 100
    top_recall = top_tp / (top_tp + top_fn + 1e-8) * 100
    top_f1 = 2 * top_precision * top_recall / (top_precision + top_recall + 1e-8)

    all_metrics['top_precision'] = float(top_precision)
    all_metrics['top_recall'] = float(top_recall)
    all_metrics['top_f1'] = float(top_f1)
    all_metrics['top_detection_rate'] = float(top_recall)

    # Overall average metrics
    valid_base = (base_predictions > 0) & (base_ground_truth > 0)
    valid_top = (top_predictions > 0) & (top_ground_truth > 0)

    if valid_base.any() and valid_top.any():
        all_errors = np.concatenate([
            np.abs(base_predictions[valid_base] - base_ground_truth[valid_base]),
            np.abs(top_predictions[valid_top] - top_ground_truth[valid_top])
        ])

        all_metrics['overall_MAE'] = float(all_errors.mean())
        all_metrics['overall_Accuracy_2m'] = float((all_errors <= 2.0).mean() * 100)

    # Overall detection metrics
    overall_precision = (base_precision + top_precision) / 2
    overall_recall = (base_recall + top_recall) / 2
    overall_f1 = (base_f1 + top_f1) / 2

    all_metrics['overall_precision'] = float(overall_precision)
    all_metrics['overall_recall'] = float(overall_recall)
    all_metrics['overall_f1'] = float(overall_f1)
    all_metrics['overall_detection_rate'] = float(overall_recall)

    return all_metrics


class MetricsTracker:
    """
    Track metrics across batches during training/evaluation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.base_preds = []
        self.base_gts = []
        self.top_preds = []
        self.top_gts = []

        self.losses = {
            'total': [],
            'seg': [],
            'reg': [],
            'conf': []
        }

    def update(
        self,
        base_pred: torch.Tensor,
        base_gt: torch.Tensor,
        top_pred: torch.Tensor,
        top_gt: torch.Tensor,
        losses: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Update tracker with batch results.

        Args:
            base_pred: [batch] predicted base distances
            base_gt: [batch] ground truth base distances
            top_pred: [batch] predicted top distances
            top_gt: [batch] ground truth top distances
            losses: Dictionary with loss values
        """
        # Convert to numpy
        self.base_preds.append(base_pred.detach().cpu().numpy())
        self.base_gts.append(base_gt.detach().cpu().numpy())
        self.top_preds.append(top_pred.detach().cpu().numpy())
        self.top_gts.append(top_gt.detach().cpu().numpy())

        # Track losses
        if losses is not None:
            for key in ['total', 'seg', 'reg', 'conf']:
                loss_key = f'{key}_loss' if key != 'total' else 'total_loss'
                if loss_key in losses:
                    self.losses[key].append(losses[loss_key].item())

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from all accumulated data.

        Returns:
            Dictionary with metrics
        """
        # Concatenate all predictions
        base_preds = np.concatenate(self.base_preds)
        base_gts = np.concatenate(self.base_gts)
        top_preds = np.concatenate(self.top_preds)
        top_gts = np.concatenate(self.top_gts)

        # Compute metrics
        metrics = compute_cliff_detection_metrics(
            base_preds, base_gts, top_preds, top_gts
        )

        # Add average losses
        for key, values in self.losses.items():
            if values:
                metrics[f'{key}_loss'] = float(np.mean(values))

        return metrics

    def get_predictions(self) -> Dict[str, np.ndarray]:
        """
        Get all predictions and ground truth.

        Returns:
            Dictionary with arrays
        """
        return {
            'base_pred': np.concatenate(self.base_preds),
            'base_gt': np.concatenate(self.base_gts),
            'top_pred': np.concatenate(self.top_preds),
            'top_gt': np.concatenate(self.top_gts)
        }


def format_metrics(metrics: Dict[str, float], indent: int = 0) -> str:
    """
    Format metrics dictionary for pretty printing.

    Args:
        metrics: Dictionary of metrics
        indent: Indentation level

    Returns:
        Formatted string
    """
    lines = []
    prefix = " " * indent

    # Group by category
    base_metrics = {k: v for k, v in metrics.items() if k.startswith('base_')}
    top_metrics = {k: v for k, v in metrics.items() if k.startswith('top_')}
    overall_metrics = {k: v for k, v in metrics.items() if k.startswith('overall_')}
    loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k}

    if base_metrics:
        lines.append(f"{prefix}Cliff Base Metrics:")
        for k, v in base_metrics.items():
            key = k.replace('base_', '  ')
            if isinstance(v, float):
                if 'Accuracy' in k or '%' in k:
                    lines.append(f"{prefix}  {key}: {v:.2f}%")
                else:
                    lines.append(f"{prefix}  {key}: {v:.4f}")
            else:
                lines.append(f"{prefix}  {key}: {v}")

    if top_metrics:
        lines.append(f"{prefix}Cliff Top Metrics:")
        for k, v in top_metrics.items():
            key = k.replace('top_', '  ')
            if isinstance(v, float):
                if 'Accuracy' in k or '%' in k:
                    lines.append(f"{prefix}  {key}: {v:.2f}%")
                else:
                    lines.append(f"{prefix}  {key}: {v:.4f}")
            else:
                lines.append(f"{prefix}  {key}: {v}")

    if overall_metrics:
        lines.append(f"{prefix}Overall Metrics:")
        for k, v in overall_metrics.items():
            key = k.replace('overall_', '  ')
            if isinstance(v, float):
                if 'Accuracy' in k or '%' in k:
                    lines.append(f"{prefix}  {key}: {v:.2f}%")
                else:
                    lines.append(f"{prefix}  {key}: {v:.4f}")

    if loss_metrics:
        lines.append(f"{prefix}Losses:")
        for k, v in loss_metrics.items():
            if isinstance(v, float):
                lines.append(f"{prefix}  {k}: {v:.6f}")

    return "\n".join(lines)
