"""Generate simulated trained-model predictions from ground-truth masks.

This script creates predictions that simulate a trained segmentation model
by starting from GT masks and adding realistic imperfections (slight
erosion/dilation, occasional missed instances).

This is for educational purposes only — in practice, predictions would
come from an actual trained model (e.g., YOLO-seg, Mask R-CNN).

Usage:
    python src/generate_trained_predictions.py \
        --gt-dir outputs/synthetic_demo/masks \
        --output-dir outputs/predictions_yolo_trained \
        --detection-rate 0.9 \
        --seed 123
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def simulate_trained_prediction(mask: np.ndarray,
                                seed: int) -> np.ndarray:
    """Apply small imperfections to a GT mask to simulate model output.

    Args:
        mask: Binary GT mask (H, W), uint8.
        seed: Random seed for reproducibility.

    Returns:
        Slightly perturbed binary mask.
    """
    rng = np.random.RandomState(seed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    result = mask.copy()
    # Randomly erode or dilate
    if rng.random() > 0.5:
        result = cv2.erode(result, kernel, iterations=1)
    else:
        result = cv2.dilate(result, kernel, iterations=1)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated trained-model predictions."
    )
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with ground-truth masks.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for predictions.")
    parser.add_argument("--detection-rate", type=float, default=0.9,
                        help="Fraction of GT instances detected (0-1).")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed.")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    label_maps = sorted(gt_dir.glob("label_map_*.npy"))
    if not label_maps:
        print(f"Error: No label maps found in {gt_dir}")
        sys.exit(1)

    print(f"Generating simulated trained-model predictions...")
    print(f"  GT dir: {gt_dir}")
    print(f"  Detection rate: {args.detection_rate}")

    for label_path in tqdm(label_maps):
        idx = int(label_path.stem.split("_")[-1])
        image_stem = f"image_{idx:04d}"

        label_map = np.load(str(label_path))
        instance_ids = np.unique(label_map)
        instance_ids = instance_ids[instance_ids > 0]

        pred_dir = output_dir / image_stem
        pred_dir.mkdir(parents=True, exist_ok=True)

        h, w = label_map.shape
        new_label_map = np.zeros((h, w), dtype=np.int32)
        mask_idx = 0

        for iid in instance_ids:
            # Simulate detection rate
            if rng.random() > args.detection_rate:
                continue

            mask = (label_map == iid).astype(np.uint8)
            pred_mask = simulate_trained_prediction(
                mask, seed=args.seed + idx * 100 + int(iid)
            )

            if pred_mask.sum() > 0:
                np.save(str(pred_dir / f"mask_{mask_idx:03d}.npy"), pred_mask)
                new_label_map[pred_mask > 0] = mask_idx + 1
                mask_idx += 1

        np.save(str(pred_dir / "label_map.npy"), new_label_map)

    print(f"Done. Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
