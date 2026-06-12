"""Generate synthetic nanosheet-like microscopy images with instance masks.

This script creates grayscale images that mimic TEM nanosheet observations.
Each nanosheet is modeled as a semi-transparent polygon with realistic noise,
blur, and contrast variations.

Usage:
    python src/generate_synthetic_nanosheets.py \
        --config configs/synthetic_mid.yaml \
        --num-images 10 \
        --output-dir outputs/synthetic_demo
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def generate_nanosheet_polygon(image_size: int,
                               size_min: int,
                               size_max: int) -> np.ndarray:
    """Generate a random polygon representing a nanosheet.

    Nanosheets are modeled as irregular quadrilaterals to hexagons.

    Args:
        image_size: Size of the canvas.
        size_min: Minimum nanosheet diameter.
        size_max: Maximum nanosheet diameter.

    Returns:
        Array of polygon vertices (N, 2).
    """
    # Random center
    cx = np.random.randint(size_max, image_size - size_max)
    cy = np.random.randint(size_max, image_size - size_max)

    # Random size
    radius = np.random.randint(size_min // 2, size_max // 2)

    # Random number of vertices (4 to 6 for nanosheet-like shape)
    n_vertices = np.random.randint(4, 7)

    # Generate vertices around center with angular perturbation
    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
    radii = radius * np.random.uniform(0.6, 1.0, n_vertices)

    vertices = np.column_stack([
        cx + radii * np.cos(angles),
        cy + radii * np.sin(angles)
    ]).astype(np.int32)

    return vertices


def render_synthetic_image(config: dict,
                           seed: int = None) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Render a single synthetic nanosheet image with instance masks.

    Args:
        config: Configuration dictionary with generation parameters.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (image, label_map, instance_masks).
        - image: grayscale (H, W), uint8
        - label_map: (H, W), int32, pixel = instance ID (0 = background)
        - instance_masks: list of binary masks (H, W), uint8
    """
    if seed is not None:
        np.random.seed(seed)

    size = config["image_size"]
    bg_intensity = config["background_intensity"]
    noise_sigma = config["noise_sigma"]
    blur_sigma = config["blur_sigma"]
    intensity_range = config["nanosheet_intensity_range"]
    overlap_prob = config["overlap_probability"]

    num_sheets = np.random.randint(
        config["num_nanosheets_min"],
        config["num_nanosheets_max"] + 1
    )

    # Create background
    image = np.full((size, size), bg_intensity, dtype=np.float32)
    label_map = np.zeros((size, size), dtype=np.int32)
    instance_masks = []

    for i in range(num_sheets):
        # Generate polygon
        poly = generate_nanosheet_polygon(
            size, config["nanosheet_size_min"], config["nanosheet_size_max"]
        )

        # Clip vertices to image bounds
        poly = np.clip(poly, 0, size - 1)

        # Create mask for this instance
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)

        # Check overlap: skip with some probability if overlapping existing
        if i > 0 and np.any(label_map[mask > 0] > 0):
            if np.random.random() > overlap_prob:
                continue

        # Random intensity for this nanosheet
        intensity = np.random.randint(intensity_range[0], intensity_range[1])

        # Apply Beer-Lambert-like attenuation (multiplicative darkening)
        attenuation = intensity / 255.0
        image[mask > 0] = image[mask > 0] * attenuation

        # Update label map
        instance_id = len(instance_masks) + 1
        label_map[mask > 0] = instance_id
        instance_masks.append(mask)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_sigma, (size, size))
    image = image + noise

    # Apply Gaussian blur
    if blur_sigma > 0:
        ksize = int(blur_sigma * 6) | 1  # Ensure odd kernel size
        image = cv2.GaussianBlur(image, (ksize, ksize), blur_sigma)

    # Clip and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image, label_map, instance_masks


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic nanosheet microscopy images."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to generate.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_images} synthetic nanosheet images...")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")

    for i in tqdm(range(args.num_images)):
        seed = args.seed + i
        image, label_map, instance_masks = render_synthetic_image(config, seed)

        # Save image
        cv2.imwrite(str(images_dir / f"image_{i:04d}.png"), image)

        # Save label map
        np.save(str(masks_dir / f"label_map_{i:04d}.npy"), label_map)

        # Save individual instance masks
        instance_dir = masks_dir / f"instances_{i:04d}"
        instance_dir.mkdir(exist_ok=True)
        for j, mask in enumerate(instance_masks):
            np.save(str(instance_dir / f"mask_{j:03d}.npy"), mask)

    print(f"Done. Generated {args.num_images} images with instance masks.")
    print(f"  Images: {images_dir}")
    print(f"  Masks:  {masks_dir}")


if __name__ == "__main__":
    main()
