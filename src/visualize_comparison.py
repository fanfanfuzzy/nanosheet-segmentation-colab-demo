#!/usr/bin/env python3
"""Visualize per-image comparison of GT, zero-shot, YOLO-seg (pretrained), and YOLO-seg (trained).

Generates a grid figure for each test image showing:
  Column 1: Original test image
  Column 2: Ground-truth instance masks
  Column 3: Zero-shot (SAM-style) predictions
  Column 4: YOLO-seg pretrained (no fine-tuning) predictions
  Column 5: YOLO-seg trained predictions

Usage:
    python src/visualize_comparison.py \
        --image-dir demo_assets/test_images \
        --gt-dir demo_assets/ground_truth \
        --zero-shot-dir demo_assets/predictions_sam_baseline \
        --pretrained-dir demo_assets/predictions_yolo_pretrained \
        --trained-dir demo_assets/predictions_yolo_trained \
        --output-dir outputs/comparison_visual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils.masks import label_map_to_instance_masks
from utils.plotting import overlay_masks_on_image
import cv2


def load_gt_masks(gt_dir: Path, idx: int) -> list[np.ndarray]:
    """Load ground-truth masks for a given image index."""
    label_path = gt_dir / f"label_map_{idx:04d}.npy"
    if label_path.exists():
        return label_map_to_instance_masks(np.load(str(label_path)))
    return []


def load_pred_masks(pred_dir: Path, image_stem: str,
                    idx: int) -> list[np.ndarray]:
    """Load predicted masks (supports both directory and flat formats)."""
    # Directory per image (zero-shot format)
    d = pred_dir / image_stem
    if d.exists():
        mask_files = sorted(d.glob("mask_*.npy"))
        if mask_files:
            return [np.load(str(f)) for f in mask_files]
        lm = d / "label_map.npy"
        if lm.exists():
            return label_map_to_instance_masks(np.load(str(lm)))

    # Flat label_map (YOLO-seg format)
    label_path = pred_dir / f"label_map_{idx:04d}.npy"
    if label_path.exists():
        return label_map_to_instance_masks(np.load(str(label_path)))

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-image comparison of segmentation methods.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory with test PNG images.")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with GT label maps.")
    parser.add_argument("--zero-shot-dir", type=str, required=True,
                        help="Directory with zero-shot predictions.")
    parser.add_argument("--pretrained-dir", type=str, default=None,
                        help="Directory with pretrained YOLO-seg predictions.")
    parser.add_argument("--trained-dir", type=str, required=True,
                        help="Directory with trained YOLO-seg predictions.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for comparison images.")
    parser.add_argument("--max-images", type=int, default=10,
                        help="Maximum number of images to visualize.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    zero_shot_dir = Path(args.zero_shot_dir)
    pretrained_dir = Path(args.pretrained_dir) if args.pretrained_dir else None
    trained_dir = Path(args.trained_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_pretrained = pretrained_dir is not None and pretrained_dir.exists()
    ncols = 5 if has_pretrained else 4

    image_paths = sorted(image_dir.glob("*.png"))[:args.max_images]
    if not image_paths:
        print(f"Error: No PNG images found in {image_dir}")
        sys.exit(1)

    print(f"Generating comparison visualizations for {len(image_paths)} images...")
    if has_pretrained:
        print("  Including YOLO-seg pretrained (base model) column.")

    for img_path in image_paths:
        idx = int(img_path.stem.split("_")[-1])
        image_stem = img_path.stem
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        gt_masks = load_gt_masks(gt_dir, idx)
        zs_masks = load_pred_masks(zero_shot_dir, image_stem, idx)
        yolo_masks = load_pred_masks(trained_dir, image_stem, idx)
        pre_masks = (load_pred_masks(pretrained_dir, image_stem, idx)
                     if has_pretrained else [])

        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original", fontsize=12)

        axes[1].imshow(overlay_masks_on_image(image, gt_masks))
        axes[1].set_title(f"Ground Truth ({len(gt_masks)})", fontsize=12)

        axes[2].imshow(overlay_masks_on_image(image, zs_masks))
        axes[2].set_title(f"SAM ({len(zs_masks)})", fontsize=12)

        if has_pretrained:
            axes[3].imshow(overlay_masks_on_image(image, pre_masks))
            axes[3].set_title(f"YOLO-seg pretrained ({len(pre_masks)})",
                              fontsize=12)
            axes[4].imshow(overlay_masks_on_image(image, yolo_masks))
            axes[4].set_title(f"YOLO-seg trained ({len(yolo_masks)})",
                              fontsize=12)
        else:
            axes[3].imshow(overlay_masks_on_image(image, yolo_masks))
            axes[3].set_title(f"YOLO-seg trained ({len(yolo_masks)})",
                              fontsize=12)

        for ax in axes:
            ax.axis("off")

        fig.suptitle(f"Test Image {idx:04d}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = output_dir / f"comparison_{idx:04d}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    # Summary grid with all images
    n = len(image_paths)
    if n > 0:
        fig, axes = plt.subplots(n, ncols, figsize=(5 * ncols, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for row, img_path in enumerate(image_paths):
            idx = int(img_path.stem.split("_")[-1])
            image_stem = img_path.stem
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            gt_masks = load_gt_masks(gt_dir, idx)
            zs_masks = load_pred_masks(zero_shot_dir, image_stem, idx)
            yolo_masks = load_pred_masks(trained_dir, image_stem, idx)
            pre_masks = (load_pred_masks(pretrained_dir, image_stem, idx)
                         if has_pretrained else [])

            axes[row, 0].imshow(image, cmap="gray")
            axes[row, 1].imshow(overlay_masks_on_image(image, gt_masks))
            axes[row, 2].imshow(overlay_masks_on_image(image, zs_masks))

            if has_pretrained:
                axes[row, 3].imshow(overlay_masks_on_image(image, pre_masks))
                axes[row, 4].imshow(overlay_masks_on_image(image, yolo_masks))
            else:
                axes[row, 3].imshow(overlay_masks_on_image(image, yolo_masks))

            axes[row, 0].set_ylabel(f"#{idx:04d}", fontsize=11, rotation=0,
                                    labelpad=30, va="center")

            if row == 0:
                axes[row, 0].set_title("Original", fontsize=12)
                axes[row, 1].set_title("Ground Truth", fontsize=12)
                axes[row, 2].set_title("SAM", fontsize=12)
                if has_pretrained:
                    axes[row, 3].set_title("YOLO pretrained", fontsize=12)
                    axes[row, 4].set_title("YOLO trained", fontsize=12)
                else:
                    axes[row, 3].set_title("YOLO-seg trained", fontsize=12)

            for col in range(ncols):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.suptitle("Per-Image Comparison: GT vs SAM vs YOLO-seg",
                     fontsize=16, fontweight="bold")
        plt.tight_layout()
        summary_path = output_dir / "comparison_summary.png"
        plt.savefig(str(summary_path), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"\nSummary grid saved: {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
