#!/usr/bin/env python3
"""
Plot training metrics from Feature Merge Block training.

Usage:
    python plot_training_metrics.py --metrics_dir ./models/train/feature_merge
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


def load_metrics(metrics_dir):
    """Load all metrics JSON files from directory."""
    metrics_files = sorted(Path(metrics_dir).glob("metrics_epoch_*.json"))

    if not metrics_files:
        print(f"❌ No metrics files found in {metrics_dir}")
        return None

    metrics = []
    for file in metrics_files:
        with open(file, "r") as f:
            metrics.append(json.load(f))

    return metrics


def plot_metrics(metrics, output_path=None):
    """Create comprehensive training metrics visualization."""
    epochs = [m["epoch"] for m in metrics]
    total_loss = [m["total_loss"] for m in metrics]
    alpha_loss = [m["alpha_loss"] for m in metrics]
    rgb_soft_loss = [m["rgb_soft_loss"] for m in metrics]
    rgb_hard_loss = [m["rgb_hard_loss"] for m in metrics]
    learning_rate = [m["learning_rate"] for m in metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Feature Merge Block Training Metrics", fontsize=16, fontweight="bold")

    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, total_loss, "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("Total Loss (Sum of All Components)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # Add best loss annotation
    best_epoch = epochs[total_loss.index(min(total_loss))]
    best_loss = min(total_loss)
    ax.axhline(
        y=best_loss,
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f"Best: {best_loss:.4f}",
    )
    ax.legend()

    # Plot 2: Component Losses
    ax = axes[0, 1]
    ax.plot(epochs, alpha_loss, "r-o", label="Alpha Loss", linewidth=2, markersize=6)
    ax.plot(
        epochs, rgb_soft_loss, "g-s", label="RGB-Soft Loss", linewidth=2, markersize=6
    )
    ax.plot(
        epochs, rgb_hard_loss, "b-^", label="RGB-Hard Loss", linewidth=2, markersize=6
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss Value", fontsize=12)
    ax.set_title("Component Losses", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # Plot 3: Loss Breakdown (Stacked Bar)
    ax = axes[1, 0]
    width = 0.6
    ax.bar(epochs, alpha_loss, width, label="Alpha", color="#ff7f0e", alpha=0.8)
    ax.bar(
        epochs,
        rgb_soft_loss,
        width,
        bottom=alpha_loss,
        label="RGB-Soft",
        color="#2ca02c",
        alpha=0.8,
    )
    bottom = [a + r for a, r in zip(alpha_loss, rgb_soft_loss)]
    ax.bar(
        epochs,
        rgb_hard_loss,
        width,
        bottom=bottom,
        label="RGB-Hard",
        color="#1f77b4",
        alpha=0.8,
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss Contribution", fontsize=12)
    ax.set_title("Loss Component Breakdown (Stacked)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Learning Rate Schedule
    ax = axes[1, 1]
    ax.plot(epochs, learning_rate, "purple", linewidth=2.5, marker="o", markersize=6)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule (Cosine)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved plot to: {output_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Training Summary Statistics")
    print("=" * 60)
    print(f"Total Epochs:        {len(epochs)}")
    print(f"Best Epoch:          {best_epoch}")
    print(f"Best Total Loss:     {best_loss:.6f}")
    print(f"Final Total Loss:    {total_loss[-1]:.6f}")
    print(
        f"Loss Improvement:    {((total_loss[0] - total_loss[-1]) / total_loss[0] * 100):.2f}%"
    )
    print(f"Final Alpha Loss:    {alpha_loss[-1]:.6f}")
    print(f"Final RGB-Soft Loss: {rgb_soft_loss[-1]:.6f}")
    print(f"Final RGB-Hard Loss: {rgb_hard_loss[-1]:.6f}")
    print(f"Final Learning Rate: {learning_rate[-1]:.2e}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Feature Merge Block training metrics"
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="./models/train/feature_merge",
        help="Directory containing metrics JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: show interactive)",
    )
    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from: {args.metrics_dir}")
    metrics = load_metrics(args.metrics_dir)

    if metrics is None:
        return

    print(f"✅ Loaded {len(metrics)} epochs of metrics")

    # Plot
    if args.output is None:
        args.output = os.path.join(args.metrics_dir, "training_metrics.png")

    plot_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
