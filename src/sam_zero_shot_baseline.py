"""Zero-shot segmentation baseline inspired by SAM-style mask proposals.

This is a lightweight educational baseline that does NOT run the actual
Segment Anything Model. Instead, it uses classical image processing
(adaptive thresholding, watershed, contour detection) to produce
candidate instance masks without any training.

The purpose is to represent a "training-free" approach for comparison
with a trained segmentation model.

Usage:
    python src/sam_zero_shot_baseline.py \
        --input-dir outputs/synthetic_demo \
        --output-dir outputs/predictions_sam_baseline
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def generate_zero_shot_masks(image: np.ndarray,
                             min_area: int = 200,
                             max_area: int = 50000) -> list[np.ndarray]:
    """Generate instance mask proposals using image processing heuristics.

    This mimics a SAM-style zero-shot approach by:
    1. Adaptive thresholding to find dark regions (nanosheets)
    2. Morphological operations to clean up noise
    3. Connected component analysis to separate instances
    4. Contour-based refinement

    Args:
        image: Grayscale input image (H, W), uint8.
        min_area: Minimum area for a valid mask.
        max_area: Maximum area for a valid mask.

    Returns:
        List of binary instance masks.
    """
    h, w = image.shape

    # Step 1: Adaptive thresholding (detect dark nanosheet regions)
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=10
    )

    # Step 2: Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 3: Distance transform for watershed seeds
    dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.3 * dist_transform.max(), 255, 0
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Step 4: Connected components on sure foreground
    num_labels, labels = cv2.connectedComponents(sure_fg)

    # Step 5: Extract individual masks
    masks = []
    for label_id in range(1, num_labels):
        mask = (labels == label_id).astype(np.uint8)
        area = mask.sum()

        if min_area <= area <= max_area:
            # Refine using the original thresholded region
            refined = cv2.bitwise_and(cleaned, cleaned, mask=mask)
            # Dilate slightly to recover edges
            refined = cv2.dilate(refined, kernel, iterations=1)
            refined = (refined > 0).astype(np.uint8)

            if refined.sum() >= min_area:
                masks.append(refined)

    return masks


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot segmentation baseline on synthetic images."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory with synthetic images.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for predictions.")
    parser.add_argument("--min-area", type=int, default=200,
                        help="Minimum mask area in pixels.")
    parser.add_argument("--max-area", type=int, default=50000,
                        help="Maximum mask area in pixels.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) / "images"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    image_files = sorted(input_dir.glob("*.png"))
    print(f"Running zero-shot baseline on {len(image_files)} images...")

    for img_path in tqdm(image_files):
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        masks = generate_zero_shot_masks(
            image, min_area=args.min_area, max_area=args.max_area
        )

        # Save predictions as label map
        stem = img_path.stem
        pred_dir = output_dir / stem
        pred_dir.mkdir(exist_ok=True)

        # Save individual masks
        for j, mask in enumerate(masks):
            np.save(str(pred_dir / f"mask_{j:03d}.npy"), mask)

        # Save combined label map
        h, w = image.shape
        label_map = np.zeros((h, w), dtype=np.int32)
        for j, mask in enumerate(masks):
            label_map[mask > 0] = j + 1
        np.save(str(pred_dir / "label_map.npy"), label_map)

    print(f"Done. Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
