"""Compare zero-shot baseline vs trained-model segmentation.

This script provides an end-to-end comparison workflow:
1. Load synthetic test images and ground-truth masks
2. Load zero-shot baseline predictions
3. Load trained-model predictions
4. Compute instance-level metrics for both methods
5. Save comparison CSV and generate a bar chart

Usage:
    python src/compare_zero_shot_vs_trained.py \
        --gt-dir outputs/synthetic_demo/masks \
        --zero-shot-dir outputs/predictions_sam_baseline \
        --trained-dir outputs/predictions_yolo_trained \
        --output-csv outputs/comparison_metrics.csv \
        --output-fig outputs/comparison_barplot.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.masks import load_instance_masks, label_map_to_instance_masks
from utils.metrics import (
    instance_recall,
    instance_precision,
    instance_f1,
    mean_best_iou,
    semantic_iou,
)
from utils.plotting import plot_metric_comparison


def load_gt_masks(masks_dir: Path, idx: int) -> list[np.ndarray]:
    """Load GT instance masks for a given image index."""
    instance_dir = masks_dir / f"instances_{idx:04d}"
    if instance_dir.exists():
        return load_instance_masks(str(instance_dir))
    label_path = masks_dir / f"label_map_{idx:04d}.npy"
    if label_path.exists():
        return label_map_to_instance_masks(np.load(str(label_path)))
    return []


def load_pred_masks(pred_dir: Path, image_stem: str) -> list[np.ndarray]:
    """Load predicted instance masks for a given image."""
    d = pred_dir / image_stem
    if d.exists():
        mask_files = sorted(d.glob("mask_*.npy"))
        return [np.load(str(f)) for f in mask_files]
    return []


def evaluate_method(gt_dir: Path,
                    pred_dir: Path,
                    method_name: str,
                    iou_threshold: float = 0.5) -> dict[str, float]:
    """Evaluate a single method across all images and return average metrics."""
    label_maps = sorted(gt_dir.glob("label_map_*.npy"))
    results = {
        "instance_recall": [],
        "instance_precision": [],
        "instance_f1": [],
        "mean_best_iou": [],
        "semantic_iou": [],
    }

    for label_path in label_maps:
        idx = int(label_path.stem.split("_")[-1])
        image_stem = f"image_{idx:04d}"

        gt_masks = load_gt_masks(gt_dir, idx)
        pred_masks = load_pred_masks(pred_dir, image_stem)

        results["instance_recall"].append(
            instance_recall(gt_masks, pred_masks, iou_threshold))
        results["instance_precision"].append(
            instance_precision(gt_masks, pred_masks, iou_threshold))
        results["instance_f1"].append(
            instance_f1(gt_masks, pred_masks, iou_threshold))
        results["mean_best_iou"].append(
            mean_best_iou(gt_masks, pred_masks))
        results["semantic_iou"].append(
            semantic_iou(gt_masks, pred_masks))

    return {k: float(np.mean(v)) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Compare zero-shot vs trained-model segmentation."
    )
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with GT masks.")
    parser.add_argument("--zero-shot-dir", type=str, required=True,
                        help="Directory with zero-shot predictions.")
    parser.add_argument("--trained-dir", type=str, required=True,
                        help="Directory with trained-model predictions.")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Output CSV for comparison.")
    parser.add_argument("--output-fig", type=str, required=True,
                        help="Output bar chart figure.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching.")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    zero_shot_dir = Path(args.zero_shot_dir)
    trained_dir = Path(args.trained_dir)

    for d, name in [(gt_dir, "GT"), (zero_shot_dir, "zero-shot"),
                     (trained_dir, "trained")]:
        if not d.exists():
            print(f"Error: {name} directory not found: {d}")
            sys.exit(1)

    n_images = len(list(gt_dir.glob("label_map_*.npy")))
    if n_images == 0:
        print("Error: No label maps found in GT directory.")
        sys.exit(1)
    print(f"Comparing zero-shot vs trained on {n_images} images...")

    # Evaluate both methods
    print("\nEvaluating zero-shot baseline...")
    zs_metrics = evaluate_method(gt_dir, zero_shot_dir, "sam_zero_shot",
                                 args.iou_threshold)

    print("Evaluating trained model...")
    tr_metrics = evaluate_method(gt_dir, trained_dir, "yolo_trained",
                                 args.iou_threshold)

    # Build comparison table
    metrics_dict = {
        "sam_zero_shot": zs_metrics,
        "yolo_trained": tr_metrics,
    }

    rows = []
    for method_name, metrics in metrics_dict.items():
        rows.append({"method": method_name, **metrics})

    comparison_df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv, index=False)
    print(f"\nComparison CSV saved to: {output_csv}")

    # Generate bar chart
    plot_metric_comparison(
        metrics_dict, args.output_fig,
        title="Zero-Shot vs Trained Model: Segmentation Metrics"
    )
    print(f"Comparison figure saved to: {args.output_fig}")

    # Print results
    print("\n" + "=" * 60)
    print("  Zero-Shot Baseline vs Trained Model Comparison")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # Print improvement summary
    print("\n--- Improvement (trained - zero-shot) ---")
    for metric in zs_metrics:
        diff = tr_metrics[metric] - zs_metrics[metric]
        arrow = "+" if diff >= 0 else ""
        print(f"  {metric}: {arrow}{diff:.4f}")


if __name__ == "__main__":
    main()
