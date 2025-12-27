"""
Multi-task loss functions for cliff detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights [background, base, top]
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha)
        else:
            self.alpha = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: [batch, seq_len, num_classes] logits
            targets: [batch, seq_len, num_classes] soft labels OR
                     [batch, seq_len] hard labels
            mask: [batch, seq_len] boolean mask (True = valid)

        Returns:
            Scalar loss
        """
        # Handle soft vs hard labels
        if targets.dim() == inputs.dim():
            # Soft labels [batch, seq_len, num_classes]
            # Convert to probability distribution
            p = F.softmax(inputs, dim=-1)
            ce_loss = -(targets * torch.log(p + 1e-8)).sum(dim=-1)  # [batch, seq_len]

            # Focal term: use max probability as p_t
            p_t = (p * targets).sum(dim=-1)  # [batch, seq_len]
            focal_weight = (1 - p_t) ** self.gamma

        else:
            # Hard labels [batch, seq_len]
            batch_size, seq_len, num_classes = inputs.shape
            inputs_flat = inputs.reshape(-1, num_classes)
            targets_flat = targets.reshape(-1)

            # Standard cross-entropy
            ce_loss = F.cross_entropy(
                inputs_flat,
                targets_flat,
                reduction='none',
                ignore_index=-100
            )
            ce_loss = ce_loss.reshape(batch_size, seq_len)

            # Focal term
            p = F.softmax(inputs, dim=-1)
            p_t = p.gather(2, targets.unsqueeze(-1).clamp(min=0)).squeeze(-1)
            focal_weight = (1 - p_t) ** self.gamma

            # Zero out ignored positions (-100 labels)
            focal_weight = focal_weight.masked_fill(targets == -100, 0.0)

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply mask if provided
        if mask is not None:
            focal_loss = focal_loss * mask.float()
            valid_count = mask.sum()
        else:
            valid_count = focal_loss.numel()

        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            if targets.dim() == inputs.dim():
                # Soft labels
                alpha_weight = (targets * alpha.view(1, 1, -1)).sum(dim=-1)
            else:
                # Hard labels
                alpha_weight = alpha[targets.clamp(min=0)]
                alpha_weight = alpha_weight.masked_fill(targets == -100, 0.0)

            focal_loss = focal_loss * alpha_weight

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_count + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CliffDetectionLoss(nn.Module):
    """
    Multi-task loss for cliff detection.

    Combines:
    1. Segmentation loss (focal loss for class imbalance)
    2. Regression loss (smooth L1 for distance offsets)
    3. Confidence loss (BCE for has_cliff prediction)
    4. Alongshore smoothness loss (optional, applied post-batch)
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_reg: float = 0.5,
        lambda_conf: float = 0.3,
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[List[float]] = None
    ):
        """
        Args:
            lambda_seg: Weight for segmentation loss
            lambda_reg: Weight for regression loss
            lambda_conf: Weight for confidence loss
            class_weights: Class weights for segmentation
            focal_gamma: Gamma parameter for focal loss
            focal_alpha: Alpha parameter for focal loss
        """
        super().__init__()

        self.lambda_seg = lambda_seg
        self.lambda_reg = lambda_reg
        self.lambda_conf = lambda_conf

        # Segmentation loss (focal loss)
        self.seg_loss_fn = FocalLoss(
            alpha=focal_alpha if focal_alpha else class_weights,
            gamma=focal_gamma
        )

        # Regression loss (smooth L1)
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none', beta=1.0)

        # Confidence loss (BCE)
        self.conf_loss_fn = nn.BCELoss()

    def forward(
        self,
        outputs: dict,
        targets: dict,
        mask: torch.Tensor
    ) -> dict:
        """
        Compute multi-task loss.

        Args:
            outputs: Dictionary from model forward():
                'segmentation': [batch, seq_len, 3]
                'regression': [batch, seq_len, 2]
                'confidence': [batch, 1]
            targets: Dictionary with:
                'seg_labels': [batch, seq_len, 3] or [batch, seq_len]
                'reg_labels': [batch, seq_len, 2]
                'base_dist_gt': [batch]
                'top_dist_gt': [batch]
            mask: [batch, seq_len] boolean mask

        Returns:
            Dictionary with:
                'total_loss': Scalar total loss
                'seg_loss': Scalar segmentation loss
                'reg_loss': Scalar regression loss
                'conf_loss': Scalar confidence loss
        """
        # 1. Segmentation loss
        seg_loss = self.seg_loss_fn(
            outputs['segmentation'],
            targets['seg_labels'],
            mask=mask
        )

        # 2. Regression loss (only on valid masked regions)
        reg_pred = outputs['regression']  # [batch, seq_len, 2]
        reg_target = targets['reg_labels']  # [batch, seq_len, 2]

        reg_loss_per_point = self.reg_loss_fn(reg_pred, reg_target)  # [batch, seq_len, 2]
        reg_loss_masked = reg_loss_per_point * mask.unsqueeze(-1).float()

        reg_loss = reg_loss_masked.sum() / (mask.sum() * 2 + 1e-8)

        # 3. Confidence loss (transect-level)
        # Ground truth: has cliff if base_dist_gt and top_dist_gt are valid (> 0)
        has_cliff_gt = (targets['base_dist_gt'] > 0) & (targets['top_dist_gt'] > 0)
        has_cliff_gt = has_cliff_gt.float().unsqueeze(-1)  # [batch, 1]

        conf_pred = outputs['confidence']  # [batch, 1]
        conf_loss = self.conf_loss_fn(conf_pred, has_cliff_gt)

        # Total loss
        total_loss = (
            self.lambda_seg * seg_loss +
            self.lambda_reg * reg_loss +
            self.lambda_conf * conf_loss
        )

        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'reg_loss': reg_loss,
            'conf_loss': conf_loss
        }


class AlongshoreSmoothness Loss(nn.Module):
    """
    Alongshore smoothness constraint.

    Penalizes large differences in predicted cliff positions
    between adjacent transects (spatial consistency).
    """

    def __init__(self, k: int = 3):
        """
        Args:
            k: Number of neighbors to consider
        """
        super().__init__()
        self.k = k

    def forward(
        self,
        base_distances: torch.Tensor,
        top_distances: torch.Tensor,
        transect_ids: List[int]
    ) -> torch.Tensor:
        """
        Args:
            base_distances: [batch] predicted base distances
            top_distances: [batch] predicted top distances
            transect_ids: List[int] transect IDs for sorting

        Returns:
            Scalar smoothness loss
        """
        # Convert transect_ids to tensor
        transect_ids_tensor = torch.LongTensor(transect_ids).to(base_distances.device)

        # Sort by transect ID (alongshore order)
        sorted_idx = torch.argsort(transect_ids_tensor)

        base_sorted = base_distances[sorted_idx]
        top_sorted = top_distances[sorted_idx]

        # Compute differences with k-nearest neighbors
        smoothness_loss = 0.0
        count = 0

        for offset in range(1, self.k + 1):
            if len(base_sorted) <= offset:
                break

            # Base differences
            base_diff = torch.abs(base_sorted[offset:] - base_sorted[:-offset])
            # Filter out invalid predictions (-1)
            valid_base = (base_sorted[offset:] > 0) & (base_sorted[:-offset] > 0)
            if valid_base.any():
                smoothness_loss += base_diff[valid_base].mean()
                count += 1

            # Top differences
            top_diff = torch.abs(top_sorted[offset:] - top_sorted[:-offset])
            valid_top = (top_sorted[offset:] > 0) & (top_sorted[:-offset] > 0)
            if valid_top.any():
                smoothness_loss += top_diff[valid_top].mean()
                count += 1

        if count > 0:
            smoothness_loss = smoothness_loss / count

        return smoothness_loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for direct distance prediction.

    Alternative to segmentation+regression approach.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [batch] predicted distances
            target: [batch] ground truth distances
            weights: [batch] optional sample weights

        Returns:
            Scalar loss
        """
        mse = (pred - target) ** 2

        if weights is not None:
            mse = mse * weights

        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse
