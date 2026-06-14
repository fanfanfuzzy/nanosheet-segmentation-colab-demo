# Demo Assets

Pre-computed data for the Colab notebook demo. All data is synthetic — no real experimental images are included.

## Contents

### `test_images/`
10 synthetic nanosheet images (512×512, PNG) used as the shared test set for evaluating both methods.

### `ground_truth/`
Ground-truth instance label maps (`label_map_XXXX.npy`, int32) for the 10 test images.
Each pixel value is an instance ID (0 = background).

### `predictions_sam/`
SAM (Segment Anything Model, ViT-H) predictions using Automatic Mask Generator (AMG) mode.
Saved as flat `label_map_XXXX.npy` files. Background masks (>50% of image area) are filtered out.
SAM is a foundation model — no task-specific fine-tuning was performed.

### `predictions_yolo_pretrained/`
YOLO-seg (YOLOv11s-seg) predictions using the **base pretrained model** (COCO weights only,
no fine-tuning on nanosheet data). This serves as a baseline showing what a generic model detects.

### `predictions_yolo_trained/`
YOLO-seg (YOLOv11s-seg) predictions on the 10 test images.
Saved as flat `label_map_XXXX.npy` files. The model was trained on 100 synthetic nanosheet images
for 150 epochs (early stopped at epoch 83, best epoch 53) on an NVIDIA RTX A6000.

## Data Policy

All assets are generated from synthetic data using the Beer-Lambert attenuation model.
No real experimental TEM images, private annotations, or unpublished data are included.
