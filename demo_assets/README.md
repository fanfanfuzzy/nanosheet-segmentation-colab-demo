# Demo Assets

Pre-computed data for the Colab notebook demo. All data is synthetic — no real experimental images are included.

## Contents

### `test_images/`
10 synthetic nanosheet images (512×512, PNG) used as the shared test set for evaluating both methods.

### `ground_truth/`
Ground-truth instance label maps (`label_map_XXXX.npy`, int32) for the 10 test images.
Each pixel value is an instance ID (0 = background).

### `predictions_sam_baseline/`
Zero-shot baseline predictions (adaptive thresholding + morphology + connected components).
Each image has a subdirectory with individual mask files and a combined label map.

### `predictions_yolo_trained/`
YOLO-seg (YOLOv11s-seg) predictions on the 10 test images.
Saved as flat `label_map_XXXX.npy` files. The model was trained on 50 synthetic nanosheet images
for 150 epochs (early stopped at epoch 40) on an NVIDIA RTX A6000.

## Data Policy

All assets are generated from synthetic data using the Beer-Lambert attenuation model.
No real experimental TEM images, private annotations, or unpublished data are included.
