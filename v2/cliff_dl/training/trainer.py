"""
Training loop for cliff detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from ..models.losses import CliffDetectionLoss
from ..models.metrics import MetricsTracker, format_metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class CliffDetectionTrainer:
    """
    Trainer for cliff detection model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: 'cuda' or 'cpu'
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = CliffDetectionLoss(
            lambda_seg=config['loss']['lambda_seg'],
            lambda_reg=config['loss']['lambda_reg'],
            lambda_conf=config['loss']['lambda_conf'],
            lambda_elev_penalty=config['loss'].get('lambda_elev_penalty', 0.2),
            lambda_width_penalty=config['loss'].get('lambda_width_penalty', 0.15),
            class_weights=config['loss']['class_weights'],
            focal_gamma=config['loss']['focal_gamma'],
            focal_alpha=config['loss'].get('focal_alpha', None),
            max_base_elevation=config['loss'].get('max_base_elevation', 15.0),
            max_cliff_width=config['loss'].get('max_cliff_width', 150.0),
            typical_cliff_width=config['loss'].get('typical_cliff_width', 100.0)
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=config['training']['scheduler']['mode'],
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor'],
            min_lr=config['training']['scheduler']['min_lr']
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode']
        )

        # Logging
        self.log_dir = Path(log_dir) if log_dir else Path('experiments/runs/logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('experiments/runs/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_metrics_tracker = MetricsTracker()
        self.val_metrics_tracker = MetricsTracker()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.train_metrics_tracker.reset()

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')

        for batch in progress_bar:
            # Move batch to device
            features = batch['features'].to(self.device)
            seg_labels = batch['seg_labels'].to(self.device)
            reg_labels = batch['reg_labels'].to(self.device)
            distances = batch['distances'].to(self.device)
            mask = batch['mask'].to(self.device)
            lengths = batch['lengths']
            base_dist_gt = batch['base_dist_gt'].to(self.device)
            top_dist_gt = batch['top_dist_gt'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features, mask=mask, lengths=lengths)

            # Compute loss
            targets = {
                'seg_labels': seg_labels,
                'reg_labels': reg_labels,
                'base_dist_gt': base_dist_gt,
                'top_dist_gt': top_dist_gt,
                # PHASE 3: Add features and distances for geomorphological penalties
                'features': features,
                'distances': distances
            }
            losses = self.criterion(outputs, targets, mask)

            # Backward pass
            losses['total_loss'].backward()

            # Gradient clipping
            if 'grad_clip' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

            self.optimizer.step()

            # Get predictions for metrics
            predictions = self.model.predict_cliff_positions(
                features, distances, mask=mask, lengths=lengths,
                confidence_threshold=self.config['postprocess']['confidence_threshold']
            )

            # Update metrics tracker
            self.train_metrics_tracker.update(
                predictions['base_distances'],
                base_dist_gt,
                predictions['top_distances'],
                top_dist_gt,
                losses
            )

            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total_loss'].item(),
                'seg': losses['seg_loss'].item(),
                'reg': losses['reg_loss'].item()
            })

        # Compute epoch metrics
        metrics = self.train_metrics_tracker.compute()
        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_metrics_tracker.reset()

        progress_bar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')

        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                features = batch['features'].to(self.device)
                seg_labels = batch['seg_labels'].to(self.device)
                reg_labels = batch['reg_labels'].to(self.device)
                distances = batch['distances'].to(self.device)
                mask = batch['mask'].to(self.device)
                lengths = batch['lengths']
                base_dist_gt = batch['base_dist_gt'].to(self.device)
                top_dist_gt = batch['top_dist_gt'].to(self.device)

                # Forward pass
                outputs = self.model(features, mask=mask, lengths=lengths)

                # Compute loss
                targets = {
                    'seg_labels': seg_labels,
                    'reg_labels': reg_labels,
                    'base_dist_gt': base_dist_gt,
                    'top_dist_gt': top_dist_gt,
                    # PHASE 3: Add features and distances for geomorphological penalties
                    'features': features,
                    'distances': distances
                }
                losses = self.criterion(outputs, targets, mask)

                # Get predictions
                predictions = self.model.predict_cliff_positions(
                    features, distances, mask=mask, lengths=lengths,
                    confidence_threshold=self.config['postprocess']['confidence_threshold']
                )

                # Update metrics
                self.val_metrics_tracker.update(
                    predictions['base_distances'],
                    base_dist_gt,
                    predictions['top_distances'],
                    top_dist_gt,
                    losses
                )

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': losses['total_loss'].item()
                })

        # Compute epoch metrics
        metrics = self.val_metrics_tracker.compute()
        return metrics

    def train(self, num_epochs: int):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print("=" * 80)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("Train Metrics:")
            print(format_metrics(train_metrics, indent=2))
            print("\nValidation Metrics:")
            print(format_metrics(val_metrics, indent=2))
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 80)

            # Save checkpoint if best
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"✓ Saved best model (val_loss: {self.best_val_loss:.6f})")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', val_metrics)

            # Early stopping
            if self.early_stopping(val_metrics['total_loss']):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Best model saved to: {self.checkpoint_dir / 'best_model.pth'}")
        print("=" * 80)

        self.writer.close()

    def _log_metrics(self, train_metrics: dict, val_metrics: dict):
        """Log metrics to TensorBoard."""
        epoch = self.current_epoch

        # Log losses
        self.writer.add_scalars('Loss/total', {
            'train': train_metrics.get('total_loss', 0),
            'val': val_metrics.get('total_loss', 0)
        }, epoch)

        self.writer.add_scalars('Loss/segmentation', {
            'train': train_metrics.get('seg_loss', 0),
            'val': val_metrics.get('seg_loss', 0)
        }, epoch)

        self.writer.add_scalars('Loss/regression', {
            'train': train_metrics.get('reg_loss', 0),
            'val': val_metrics.get('reg_loss', 0)
        }, epoch)

        # Log metrics
        for prefix in ['base_', 'top_']:
            for metric in ['MAE', 'RMSE', 'R2']:
                key = f'{prefix}{metric}'
                if key in train_metrics and key in val_metrics:
                    self.writer.add_scalars(f'Metrics/{key}', {
                        'train': train_metrics[key],
                        'val': val_metrics[key]
                    }, epoch)

        # Log learning rate
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

    def save_checkpoint(self, filename: str, metrics: dict):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
