#!/usr/bin/env python3
"""
Evaluation Script for CliffDelineaTool v2.0

Evaluate trained model on test set and generate comprehensive metrics.

Usage:
    python 03_evaluate_model.py --checkpoint ../experiments/runs/checkpoints/best_model.pth
    python 03_evaluate_model.py --checkpoint best_model.pth --test_data ../data/preprocessed/test.pt
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cliff_dl.data.dataset import CliffTransectDataset
from cliff_dl.data.loaders import create_data_loader
from cliff_dl.models.cnn_lstm import CNN_BiLSTM_CliffDetector
from cliff_dl.models.metrics import compute_cliff_detection_metrics, format_metrics
from cliff_dl.inference.postprocess import (
    predictions_to_dataframe,
    postprocess_predictions
)
from cliff_dl.inference.export import export_to_csv
import matplotlib.pyplot as plt


def evaluate_model(
    model,
    data_loader,
    device,
    config,
    apply_postprocessing=True
):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to use
        config: Configuration dictionary
        apply_postprocessing: Whether to apply post-processing

    Returns:
        Dictionary with predictions and metrics
    """
    model.eval()

    all_transect_ids = []
    all_base_pred = []
    all_base_gt = []
    all_top_pred = []
    all_top_gt = []
    all_base_conf = []
    all_top_conf = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Move to device
            features = batch['features'].to(device)
            distances = batch['distances'].to(device)
            mask = batch['mask'].to(device)
            lengths = batch['lengths']

            # Get predictions
            predictions = model.predict_cliff_positions(
                features, distances, mask=mask, lengths=lengths,
                confidence_threshold=config['postprocess']['confidence_threshold']
            )

            # Collect results
            all_transect_ids.extend(batch['transect_ids'])
            all_base_pred.append(predictions['base_distances'].cpu().numpy())
            all_top_pred.append(predictions['top_distances'].cpu().numpy())
            all_base_conf.append(predictions['base_confidences'].cpu().numpy())
            all_top_conf.append(predictions['top_confidences'].cpu().numpy())
            all_base_gt.append(batch['base_dist_gt'].numpy())
            all_top_gt.append(batch['top_dist_gt'].numpy())

    # Concatenate all batches
    base_pred = np.concatenate(all_base_pred)
    top_pred = np.concatenate(all_top_pred)
    base_conf = np.concatenate(all_base_conf)
    top_conf = np.concatenate(all_top_conf)
    base_gt = np.concatenate(all_base_gt)
    top_gt = np.concatenate(all_top_gt)

    # Create predictions dataframe
    predictions_df = predictions_to_dataframe(
        all_transect_ids,
        base_pred,
        top_pred,
        base_conf,
        top_conf
    )

    # Apply post-processing if requested
    if apply_postprocessing:
        print("\nApplying post-processing...")
        predictions_df_clean = postprocess_predictions(
            predictions_df,
            smooth_window=config['postprocess']['smooth_window'],
            outlier_threshold=config['postprocess']['outlier_std_threshold'],
            min_base_top_distance=config['postprocess']['min_base_top_distance']
        )

        # Update predictions with cleaned values
        base_pred_clean = predictions_df_clean['base_distance'].values
        top_pred_clean = predictions_df_clean['top_distance'].values
    else:
        base_pred_clean = base_pred
        top_pred_clean = top_pred

    # Compute metrics
    print("\nComputing metrics...")
    raw_metrics = compute_cliff_detection_metrics(base_pred, base_gt, top_pred, top_gt)
    raw_metrics = {f'raw_{k}': v for k, v in raw_metrics.items()}

    if apply_postprocessing:
        clean_metrics = compute_cliff_detection_metrics(
            base_pred_clean, base_gt, top_pred_clean, top_gt
        )
        clean_metrics = {f'clean_{k}': v for k, v in clean_metrics.items()}
        metrics = {**raw_metrics, **clean_metrics}
    else:
        metrics = raw_metrics

    results = {
        'predictions_df': predictions_df_clean if apply_postprocessing else predictions_df,
        'base_pred': base_pred_clean if apply_postprocessing else base_pred,
        'top_pred': top_pred_clean if apply_postprocessing else top_pred,
        'base_gt': base_gt,
        'top_gt': top_gt,
        'metrics': metrics
    }

    return results


def plot_error_distribution(results, output_dir):
    """
    Plot error distribution histograms.

    Args:
        results: Results dictionary from evaluate_model
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute errors
    base_pred = results['base_pred']
    base_gt = results['base_gt']
    top_pred = results['top_pred']
    top_gt = results['top_gt']

    valid_base = (base_pred > 0) & (base_gt > 0)
    valid_top = (top_pred > 0) & (top_gt > 0)

    base_errors = base_pred[valid_base] - base_gt[valid_base]
    top_errors = top_pred[valid_top] - top_gt[valid_top]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Base error histogram
    axes[0, 0].hist(base_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Error (m)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Cliff Base Error Distribution (MAE={np.abs(base_errors).mean():.2f}m)')
    axes[0, 0].grid(alpha=0.3)

    # Top error histogram
    axes[0, 1].hist(top_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Error (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Cliff Top Error Distribution (MAE={np.abs(top_errors).mean():.2f}m)')
    axes[0, 1].grid(alpha=0.3)

    # Base scatter plot
    axes[1, 0].scatter(base_gt[valid_base], base_pred[valid_base], alpha=0.5, s=10)
    min_val = min(base_gt[valid_base].min(), base_pred[valid_base].min())
    max_val = max(base_gt[valid_base].max(), base_pred[valid_base].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('Ground Truth (m)')
    axes[1, 0].set_ylabel('Predicted (m)')
    axes[1, 0].set_title('Cliff Base: Predicted vs Ground Truth')
    axes[1, 0].grid(alpha=0.3)

    # Top scatter plot
    axes[1, 1].scatter(top_gt[valid_top], top_pred[valid_top], alpha=0.5, s=10, color='orange')
    min_val = min(top_gt[valid_top].min(), top_pred[valid_top].min())
    max_val = max(top_gt[valid_top].max(), top_pred[valid_top].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Ground Truth (m)')
    axes[1, 1].set_ylabel('Predicted (m)')
    axes[1, 1].set_title('Cliff Top: Predicted vs Ground Truth')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_dir / 'error_distribution.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cliff detection model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='../data/preprocessed/test.pt',
        help='Path to test data .pt file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../outputs/evaluation',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--no_postprocessing',
        action='store_true',
        help='Disable post-processing'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda or cpu'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CliffDelineaTool v2.0 - Model Evaluation")
    print("=" * 80)

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

    # Load test data
    print(f"\nLoading test data: {args.test_data}")
    test_dataset = CliffTransectDataset(
        args.test_data,
        transform=None,
        use_soft_labels=config.get('data', {}).get('use_soft_labels', True)
    )

    test_loader = create_data_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )

    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Test batches: {len(test_loader)}")

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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    results = evaluate_model(
        model,
        test_loader,
        device,
        config,
        apply_postprocessing=not args.no_postprocessing
    )

    # Print metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(format_metrics(results['metrics']))
    print("=" * 80)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_path = output_dir / 'predictions.csv'
    export_to_csv(results['predictions_df'], predictions_path)

    # Save metrics
    metrics_path = output_dir / 'metrics.csv'
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to: {metrics_path}")

    # Plot error distribution
    print("\nGenerating plots...")
    plot_error_distribution(results, output_dir)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"  - predictions.csv: Predicted cliff positions")
    print(f"  - metrics.csv: Evaluation metrics")
    print(f"  - error_distribution.png: Error distribution plots")
    print("=" * 80)


if __name__ == '__main__':
    main()
