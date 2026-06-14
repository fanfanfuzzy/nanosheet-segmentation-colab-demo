"""Evaluate segmentation predictions against ground-truth masks.

Computes instance-level metrics and saves results to CSV.

Usage:
    python src/evaluate_predictions.py \
        --gt-dir outputs/synthetic_demo/masks \
        --pred-dir demo_assets/predictions_sam \
        --method-name sam_vit_h \
        --output outputs/metrics_sam.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.masks import load_instance_masks, label_map_to_instance_masks
from utils.metrics import (
    instance_recall,
    instance_precision,
    instance_f1,
    mean_best_iou,
    semantic_iou,
)


def load_gt_masks_for_image(masks_dir: Path, image_idx: int) -> list[np.ndarray]:
    """Load GT instance masks for a specific image."""
    instance_dir = masks_dir / f"instances_{image_idx:04d}"
    if instance_dir.exists():
        return load_instance_masks(str(instance_dir))

    # Fallback: load from label map
    label_path = masks_dir / f"label_map_{image_idx:04d}.npy"
    if label_path.exists():
        label_map = np.load(str(label_path))
        return label_map_to_instance_masks(label_map)

    return []


def load_pred_masks_for_image(pred_dir: Path, image_stem: str,
                              idx: int = 0) -> list[np.ndarray]:
    """Load predicted instance masks for a specific image."""
    instance_dir = pred_dir / image_stem
    if instance_dir.exists():
        # Load individual mask files
        mask_files = sorted(instance_dir.glob("mask_*.npy"))
        if mask_files:
            return [np.load(str(f)) for f in mask_files]
        # Check for label_map inside directory
        lm = instance_dir / "label_map.npy"
        if lm.exists():
            return label_map_to_instance_masks(np.load(str(lm)))

    # Fallback: flat label_map file (e.g. from YOLO-seg inference)
    label_path = pred_dir / f"label_map_{idx:04d}.npy"
    if label_path.exists():
        return label_map_to_instance_masks(np.load(str(label_path)))

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate instance segmentation predictions."
    )
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with ground-truth masks.")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory with predicted masks.")
    parser.add_argument("--method-name", type=str, default="method",
                        help="Name of the method (for CSV output).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching.")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    if not gt_dir.exists():
        print(f"Error: GT directory not found: {gt_dir}")
        sys.exit(1)
    if not pred_dir.exists():
        print(f"Error: Prediction directory not found: {pred_dir}")
        sys.exit(1)

    # Find all GT label maps
    label_maps = sorted(gt_dir.glob("label_map_*.npy"))
    if not label_maps:
        print("Error: No label maps found in GT directory.")
        sys.exit(1)

    print(f"Evaluating {len(label_maps)} images...")
    print(f"  Method: {args.method_name}")
    print(f"  IoU threshold: {args.iou_threshold}")

    results = []
    for label_path in tqdm(label_maps):
        # Extract image index from filename
        stem = label_path.stem  # e.g., "label_map_0000"
        idx = int(stem.split("_")[-1])
        image_stem = f"image_{idx:04d}"

        gt_masks = load_gt_masks_for_image(gt_dir, idx)
        pred_masks = load_pred_masks_for_image(pred_dir, image_stem, idx)

        # Compute metrics
        rec = instance_recall(gt_masks, pred_masks, args.iou_threshold)
        prec = instance_precision(gt_masks, pred_masks, args.iou_threshold)
        f1 = instance_f1(gt_masks, pred_masks, args.iou_threshold)
        mbi = mean_best_iou(gt_masks, pred_masks)
        siou = semantic_iou(gt_masks, pred_masks)

        results.append({
            "image": image_stem,
            "method": args.method_name,
            "num_gt": len(gt_masks),
            "num_pred": len(pred_masks),
            "instance_recall": rec,
            "instance_precision": prec,
            "instance_f1": f1,
            "mean_best_iou": mbi,
            "semantic_iou": siou,
        })

    # Create DataFrame and save
    df = pd.DataFrame(results)

    # Add summary row with averages
    summary = {
        "image": "AVERAGE",
        "method": args.method_name,
        "num_gt": df["num_gt"].mean(),
        "num_pred": df["num_pred"].mean(),
        "instance_recall": df["instance_recall"].mean(),
        "instance_precision": df["instance_precision"].mean(),
        "instance_f1": df["instance_f1"].mean(),
        "mean_best_iou": df["mean_best_iou"].mean(),
        "semantic_iou": df["semantic_iou"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"\n--- Average Metrics ({args.method_name}) ---")
    print(f"  instance_recall:    {summary['instance_recall']:.4f}")
    print(f"  instance_precision: {summary['instance_precision']:.4f}")
    print(f"  instance_f1:        {summary['instance_f1']:.4f}")
    print(f"  mean_best_iou:      {summary['mean_best_iou']:.4f}")
    print(f"  semantic_iou:       {summary['semantic_iou']:.4f}")


if __name__ == "__main__":
    main()
