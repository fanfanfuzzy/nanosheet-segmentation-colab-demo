# Demo Assets

Pre-computed data for the Colab notebook demo. All data is synthetic — no real experimental images are included.

## Directory Structure

Assets are organized by difficulty level (`easy/`, `mid/`, `hard/`), each containing:

```
demo_assets/{difficulty}/
├── test_images/              10 synthetic nanosheet images (512x512, PNG)
├── ground_truth/             GT instance label maps (label_map_XXXX.npy, int32)
├── predictions_sam/          SAM (ViT-H) AMG predictions
├── predictions_yolo_pretrained/  YOLO-seg pretrained (COCO weights, no fine-tuning)
└── predictions_yolo_trained/     YOLO-seg trained on 100 synthetic images per difficulty
```

## Difficulty Levels

Difficulty presets follow `2603-nanosheet-overlap-segmentation` definitions:

| Preset | noise_scale | alpha (contrast) | boundary_visibility | SNR |
|--------|-------------|-------------------|---------------------|-----|
| `realistic_easy` | 0.10–0.25 | 0.15–0.30 | 0.80–0.95 | ~2.7 |
| `realistic_mid`  | 0.20–0.40 | 0.10–0.20 | 0.70–0.90 | ~1.7 |
| `realistic_hard` | 0.40–0.70 | 0.08–0.18 | 0.60–0.95 | ~1.1 |

## Prediction Methods

### SAM (ViT-H) AMG
Segment Anything Model with Automatic Mask Generator mode.
Background masks (>50% of image area) are filtered out.
SAM is a foundation model — no task-specific fine-tuning was performed.

### YOLO-seg pretrained
YOLOv11s-seg with COCO pretrained weights only (no fine-tuning on nanosheet data).
Serves as a baseline showing that general-purpose models cannot segment nanosheets.

### YOLO-seg trained
YOLOv11s-seg fine-tuned on 100 synthetic nanosheet images per difficulty level.
Trained for 150 epochs with early stopping (patience=30) on NVIDIA RTX A6000.

## Data Policy

All assets are generated from synthetic data using the Beer-Lambert attenuation model.
No real experimental TEM images, private annotations, or unpublished data are included.
