"""Evaluation metrics for instance segmentation.

Metrics implemented:
- instance_recall: fraction of GT instances matched by predictions
- instance_precision: fraction of predicted instances matching a GT
- instance_f1: harmonic mean of recall and precision
- mean_best_iou: average of the best IoU for each GT instance
- semantic_iou: pixel-level IoU ignoring instance boundaries
"""

import numpy as np
from .masks import compute_iou


def instance_recall(gt_masks: list[np.ndarray],
                    pred_masks: list[np.ndarray],
                    iou_threshold: float = 0.5) -> float:
    """Compute instance recall.

    For each ground-truth instance, check if any predicted mask
    has IoU >= threshold. Recall = matched GT / total GT.

    Args:
        gt_masks: List of ground-truth binary masks.
        pred_masks: List of predicted binary masks.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        Recall score between 0 and 1.
    """
    if len(gt_masks) == 0:
        return 1.0
    if len(pred_masks) == 0:
        return 0.0

    matched = 0
    for gt in gt_masks:
        best_iou = max(compute_iou(gt, pred) for pred in pred_masks)
        if best_iou >= iou_threshold:
            matched += 1
    return matched / len(gt_masks)


def instance_precision(gt_masks: list[np.ndarray],
                       pred_masks: list[np.ndarray],
                       iou_threshold: float = 0.5) -> float:
    """Compute instance precision.

    For each predicted mask, check if any GT mask has IoU >= threshold.
    Precision = matched predictions / total predictions.

    Args:
        gt_masks: List of ground-truth binary masks.
        pred_masks: List of predicted binary masks.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        Precision score between 0 and 1.
    """
    if len(pred_masks) == 0:
        return 1.0
    if len(gt_masks) == 0:
        return 0.0

    matched = 0
    for pred in pred_masks:
        best_iou = max(compute_iou(gt, pred) for gt in gt_masks)
        if best_iou >= iou_threshold:
            matched += 1
    return matched / len(pred_masks)


def instance_f1(gt_masks: list[np.ndarray],
                pred_masks: list[np.ndarray],
                iou_threshold: float = 0.5) -> float:
    """Compute instance F1 score (harmonic mean of recall and precision).

    Args:
        gt_masks: List of ground-truth binary masks.
        pred_masks: List of predicted binary masks.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        F1 score between 0 and 1.
    """
    rec = instance_recall(gt_masks, pred_masks, iou_threshold)
    prec = instance_precision(gt_masks, pred_masks, iou_threshold)
    if rec + prec == 0:
        return 0.0
    return 2 * rec * prec / (rec + prec)


def mean_best_iou(gt_masks: list[np.ndarray],
                  pred_masks: list[np.ndarray]) -> float:
    """Compute mean of the best IoU for each GT instance.

    For each GT mask, find the predicted mask with highest IoU.
    Return the average of these best IoU values.

    Args:
        gt_masks: List of ground-truth binary masks.
        pred_masks: List of predicted binary masks.

    Returns:
        Mean best IoU between 0 and 1.
    """
    if len(gt_masks) == 0:
        return 1.0
    if len(pred_masks) == 0:
        return 0.0

    best_ious = []
    for gt in gt_masks:
        best = max(compute_iou(gt, pred) for pred in pred_masks)
        best_ious.append(best)
    return float(np.mean(best_ious))


def semantic_iou(gt_masks: list[np.ndarray],
                 pred_masks: list[np.ndarray]) -> float:
    """Compute semantic (pixel-level) IoU.

    Merge all GT masks into one foreground map and all predicted masks
    into another, then compute their IoU. This ignores instance boundaries.

    Args:
        gt_masks: List of ground-truth binary masks.
        pred_masks: List of predicted binary masks.

    Returns:
        Semantic IoU between 0 and 1.
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0

    # Create combined foreground maps
    if len(gt_masks) > 0:
        h, w = gt_masks[0].shape
    else:
        h, w = pred_masks[0].shape

    gt_fg = np.zeros((h, w), dtype=bool)
    for m in gt_masks:
        gt_fg |= (m > 0)

    pred_fg = np.zeros((h, w), dtype=bool)
    for m in pred_masks:
        pred_fg |= (m > 0)

    intersection = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
