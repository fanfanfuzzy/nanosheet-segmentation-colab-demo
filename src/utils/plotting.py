"""Visualization utilities for nanosheet segmentation demo."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


def get_instance_colormap(num_instances: int) -> ListedColormap:
    """Generate a colormap with distinct colors for instances."""
    if num_instances == 0:
        return ListedColormap(["black"])
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_instances, 2)))
    return ListedColormap(colors)


def overlay_masks_on_image(image: np.ndarray,
                           masks: list[np.ndarray],
                           alpha: float = 0.4) -> np.ndarray:
    """Overlay colored instance masks on a grayscale image.

    Args:
        image: Grayscale image (H, W), values in [0, 255].
        masks: List of binary masks.
        alpha: Transparency for the overlay.

    Returns:
        RGB image (H, W, 3) with colored masks overlaid.
    """
    h, w = image.shape[:2]
    rgb = np.stack([image] * 3, axis=-1).astype(np.float32)
    if rgb.max() > 1:
        rgb = rgb / 255.0

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(masks), 1)))

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)][:3]
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask > 0,
                rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
                rgb[:, :, c]
            )
    return (rgb * 255).astype(np.uint8)


def plot_metric_comparison(metrics_dict: dict[str, dict[str, float]],
                           output_path: str,
                           title: str = "Segmentation Metrics Comparison") -> None:
    """Create a grouped bar chart comparing metrics across methods.

    Args:
        metrics_dict: {method_name: {metric_name: value}}.
        output_path: Path to save the figure.
        title: Figure title.
    """
    methods = list(metrics_dict.keys())
    if not methods:
        return

    metric_names = list(metrics_dict[methods[0]].keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = [metrics_dict[method].get(m, 0) for m in metric_names]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_overview(images: list[np.ndarray],
                          masks_list: list[list[np.ndarray]],
                          output_path: str,
                          max_samples: int = 4) -> None:
    """Plot a grid showing images and their instance masks.

    Args:
        images: List of grayscale images.
        masks_list: List of mask lists (one list per image).
        output_path: Path to save the figure.
        max_samples: Maximum samples to show.
    """
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(images[i], cmap="gray")
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")

        overlay = overlay_masks_on_image(images[i], masks_list[i])
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"GT Masks ({len(masks_list[i])} instances)")
        axes[1, i].axis("off")

    plt.suptitle("Synthetic Nanosheet Dataset", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
