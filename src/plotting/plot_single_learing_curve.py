#!/usr/bin/env python3
"""
Plot learning curve from a single pkl file containing training data.

Usage:
    python plot_single_learning_curve.py <path_to_pkl_file> [options]

Example:
    python plot_single_learning_curve.py logs/train.dat --key rewards_per_episode --smooth 10
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w):
    """Compute moving average with window size w."""
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w), 'valid') / w


def load_pkl_data(pkl_path, key=None):
    """
    Load data from pkl file.

    Args:
        pkl_path: Path to pkl file
        key: Optional key to extract from dictionary. If None, will try common keys.

    Returns:
        data: numpy array of training rewards/metrics
        key_used: the key that was actually used
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # If data is already a dict, extract the appropriate key
    if isinstance(data, dict):
        if key is not None:
            if key in data:
                return np.array(data[key]), key
            else:
                raise KeyError(f"Key '{key}' not found in pkl file. Available keys: {list(data.keys())}")

        # Try common keys if not specified
        common_keys = ['rewards_per_episode', 'rewards_per_iteration', 'returns', 'rewards']
        for k in common_keys:
            if k in data:
                return np.array(data[k]), k

        # If none of the common keys work, just use the first array-like value
        for k, v in data.items():
            if isinstance(v, (list, np.ndarray)):
                return np.array(v), k

        raise ValueError(f"Could not find suitable data in pkl file. Keys: {list(data.keys())}")

    # If data is already an array, just return it
    return np.array(data), "data"


def plot_learning_curve(pkl_path, output_path=None, key=None, smooth_window=1,
                       title=None, xlabel="Episode", ylabel="Reward", figsize=(12, 7)):
    """
    Plot learning curve from pkl file.

    Args:
        pkl_path: Path to pkl file
        output_path: Path to save figure (if None, uses pkl_path with .png extension)
        key: Key to extract from pickle dict
        smooth_window: Moving average window size (default: 1, no smoothing)
        title: Plot title (default: auto-generated)
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl file not found: {pkl_path}")

    # Load data
    rewards, key_used = load_pkl_data(pkl_path, key)

    # Apply smoothing if requested
    if smooth_window > 1:
        rewards_smoothed = moving_average(rewards, smooth_window)
    else:
        rewards_smoothed = rewards

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(rewards_smoothed))
    ax.plot(x, rewards_smoothed, linewidth=2)

    # Add raw data as light background if smoothing was applied
    if smooth_window > 1:
        x_raw = np.arange(len(rewards))
        ax.plot(x_raw, rewards, alpha=0.2, linewidth=0.5, label=f"Raw data")
        ax.plot(x, rewards_smoothed, linewidth=2, label=f"Moving avg (window={smooth_window})")
        ax.legend()

    # Labels and formatting
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f"Learning Curve - {key_used}"
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = pkl_path.parent / f"{pkl_path.stem}_learning_curve.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    print(f"Data shape: {rewards.shape}")
    print(f"Mean reward: {np.mean(rewards):.4f}")
    print(f"Max reward: {np.max(rewards):.4f}")
    print(f"Min reward: {np.min(rewards):.4f}")

    plt.show()

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot learning curve from pkl file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python plot_single_learning_curve.py logs/train.dat

  # With smoothing
  python plot_single_learning_curve.py logs/train.dat --smooth 10

  # Specify data key explicitly
  python plot_single_learning_curve.py logs/train.dat --key rewards_per_iteration

  # Save to custom path
  python plot_single_learning_curve.py logs/train.dat --output my_plot.png
        """
    )

    parser.add_argument('pkl_file', help='Path to pkl file with training data')
    parser.add_argument('--output', '-o', default=None, help='Output path for figure (default: auto-generated)')
    parser.add_argument('--key', '-k', default=None, help='Key to extract from pkl dict')
    parser.add_argument('--smooth', '-s', type=int, default=1, help='Moving average window size (default: 1)')
    parser.add_argument('--title', '-t', default=None, help='Plot title')
    parser.add_argument('--xlabel', default='Episode', help='X-axis label')
    parser.add_argument('--ylabel', default='Reward', help='Y-axis label')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 7], help='Figure size (width height)')

    args = parser.parse_args()

    plot_learning_curve(
        args.pkl_file,
        output_path=args.output,
        key=args.key,
        smooth_window=args.smooth,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=tuple(args.figsize)
    )