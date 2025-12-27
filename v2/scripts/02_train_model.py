#!/usr/bin/env python3
"""
Training Script for CliffDelineaTool v2.0

Train the deep learning model for cliff detection.

Usage:
    python 02_train_model.py --config ../config/default_config.yaml
    python 02_train_model.py --train_data ../data/preprocessed/train.pt --val_data ../data/preprocessed/val.pt
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cliff_dl.data.dataset import CliffTransectDataset, AugmentTransect
from cliff_dl.data.loaders import create_data_loader
from cliff_dl.models.cnn_lstm import CNN_BiLSTM_CliffDetector
from cliff_dl.training.trainer import CliffDetectionTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Train cliff detection model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--train_data',
        type=str,
        default=None,
        help='Path to training data .pt file (overrides config)'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help='Path to validation data .pt file (overrides config)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device: cuda or cpu (overrides config)'
    )

    args = parser.parse_args()

    # Load config
    print("=" * 80)
    print("CliffDelineaTool v2.0 - Training")
    print("=" * 80)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.train_data:
        train_data_path = args.train_data
    else:
        train_data_path = Path(config['paths'].get('data', '../data/preprocessed')) / 'train.pt'

    if args.val_data:
        val_data_path = args.val_data
    else:
        val_data_path = Path(config['paths'].get('data', '../data/preprocessed')) / 'val.pt'

    if args.epochs:
        config['training']['num_epochs'] = args.epochs

    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    if args.device:
        config['device'] = args.device

    # Set device
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'

    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"\nSet random seed: {seed}")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Training data: {train_data_path}")
    print(f"  Validation data: {val_data_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Number of epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")

    # Create data augmentation
    use_augmentation = config.get('data', {}).get('use_augmentation', True)
    if use_augmentation:
        transform = AugmentTransect(
            noise_std=0.1,
            shift_range=1.0,
            noise_prob=0.5,
            shift_prob=0.5
        )
        print(f"  Data augmentation: enabled")
    else:
        transform = None
        print(f"  Data augmentation: disabled")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CliffTransectDataset(
        train_data_path,
        transform=transform,
        use_soft_labels=config.get('data', {}).get('use_soft_labels', True)
    )

    val_dataset = CliffTransectDataset(
        val_data_path,
        transform=None,  # No augmentation for validation
        use_soft_labels=config.get('data', {}).get('use_soft_labels', True)
    )

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=(device == 'cuda')
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=(device == 'cuda')
    )

    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = CNN_BiLSTM_CliffDetector(
        input_dim=config['features']['input_dim'],
        cnn_channels=config['model']['cnn']['channels'],
        cnn_kernel_sizes=config['model']['cnn']['kernel_sizes'],
        cnn_dropout=config['model']['cnn']['dropout'],
        lstm_hidden_size=config['model']['lstm']['hidden_size'],
        lstm_num_layers=config['model']['lstm']['num_layers'],
        lstm_dropout=config['model']['lstm']['dropout'],
        attention_heads=config['model']['attention']['num_heads'],
        attention_dim=config['model']['attention']['dim'],
        attention_dropout=config['model']['attention']['dropout']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    log_dir = config['paths'].get('logs', 'experiments/runs/logs')
    checkpoint_dir = config['paths'].get('checkpoints', 'experiments/runs/checkpoints')

    trainer = CliffDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    print("\n" + "=" * 80)
    trainer.train(num_epochs=config['training']['num_epochs'])

    print("\nTraining complete!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"Model checkpoints saved to: {checkpoint_dir}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\nNext steps:")
    print(f"  1. Evaluate: python scripts/03_evaluate_model.py --checkpoint {checkpoint_dir}/best_model.pth")
    print(f"  2. Predict: python scripts/04_predict.py --checkpoint {checkpoint_dir}/best_model.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()
