"""Visualize synthetic nanosheet dataset.

Creates overview figures showing generated images and their instance masks.

Usage:
    python src/visualize_dataset.py \
        --input-dir outputs/synthetic_demo \
        --output-dir outputs/figures
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.masks import load_instance_masks
from utils.plotting import plot_dataset_overview, overlay_masks_on_image


def main():
    parser = argparse.ArgumentParser(
        description="Visualize synthetic nanosheet dataset."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory with synthetic data.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for figures.")
    parser.add_argument("--max-samples", type=int, default=4,
                        help="Maximum number of samples to visualize.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Load images and masks
    image_files = sorted(images_dir.glob("*.png"))[:args.max_samples]
    images = []
    masks_list = []

    for img_path in image_files:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        images.append(image)

        # Load corresponding masks
        idx = int(img_path.stem.split("_")[-1])
        instance_dir = masks_dir / f"instances_{idx:04d}"
        if instance_dir.exists():
            masks = load_instance_masks(str(instance_dir))
        else:
            masks = []
        masks_list.append(masks)

    # Plot overview
    output_path = str(output_dir / "sample_dataset.png")
    plot_dataset_overview(images, masks_list, output_path, args.max_samples)
    print(f"Dataset overview saved to: {output_path}")


if __name__ == "__main__":
    main()
