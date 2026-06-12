"""Utility functions for handling instance segmentation masks."""

import numpy as np
from pathlib import Path


def load_instance_masks(mask_dir: str) -> list[np.ndarray]:
    """Load instance masks from a directory.

    Each mask file is a .npy file containing a 2D array where
    pixel value 1 = foreground for that instance.

    Args:
        mask_dir: Directory containing .npy mask files.

    Returns:
        List of 2D binary masks (one per instance).
    """
    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob("*.npy"))
    masks = [np.load(f) for f in mask_files]
    return masks


def load_label_map(label_path: str) -> np.ndarray:
    """Load a label map where each pixel value is the instance ID.

    Args:
        label_path: Path to a .npy file with the label map.

    Returns:
        2D array of instance IDs (0 = background).
    """
    return np.load(label_path)


def label_map_to_instance_masks(label_map: np.ndarray) -> list[np.ndarray]:
    """Convert a label map to a list of binary instance masks.

    Args:
        label_map: 2D array where each pixel = instance ID (0 = background).

    Returns:
        List of binary masks (one per instance, excluding background).
    """
    instance_ids = np.unique(label_map)
    instance_ids = instance_ids[instance_ids > 0]
    masks = [(label_map == iid).astype(np.uint8) for iid in instance_ids]
    return masks


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks.

    Args:
        mask_a: Binary mask (H, W).
        mask_b: Binary mask (H, W).

    Returns:
        IoU score between 0 and 1.
    """
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)
