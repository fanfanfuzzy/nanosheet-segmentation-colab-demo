# Nanosheet Segmentation Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fanfanfuzzy/nanosheet-segmentation-colab-demo/blob/main/notebooks/nanosheet_segmentation_demo.ipynb)

**Beginner-friendly Google Colab demo for synthetic nanosheet instance segmentation.**

This workshop demonstrates the full workflow from synthetic data generation to segmentation prediction, evaluation, and visualization — comparing a zero-shot baseline with a YOLO-seg model trained on synthetic data.

## Data Policy / データ公開方針

> **This public repository does not contain real experimental TEM images, private annotations, or unpublished experimental datasets.**
>
> All images, masks, model weights, predictions, and sample metrics included in this repository are synthetic or generated for educational purposes.
> The YOLO-seg model weights (`models/yolo11s-seg-nanosheet.pt`) were trained exclusively on synthetic nanosheet images.

> **このリポジトリには、実際の実験TEM画像、手動アノテーション、未公開の実験データは含めていません。**
>
> 含まれる画像・マスク・モデルweights・予測結果は、教材用に生成した人工ナノシート画像、またはそれに基づくサンプル結果です。
> YOLO-segモデル（`models/yolo11s-seg-nanosheet.pt`）は合成画像のみで学習しています。

## What This Demo Shows / このデモで学ぶこと

This demo compares two approaches to instance segmentation:

1. **Zero-shot segmentation baseline** — Uses image processing heuristics (adaptive thresholding, morphological operations, connected components) to produce candidate masks **without any training**. This represents the idea behind ViT-based models like SAM (Segment Anything Model).

2. **YOLO-seg (trained on synthetic data)** — Uses YOLOv11s-seg trained on 100 synthetic nanosheet images with instance-level polygon labels. The pre-trained model and inference results are included in the repository. The key message: *a model trained on task-specific synthetic data can significantly improve instance-level segmentation performance compared to a training-free baseline.*

The dedicated comparison script (`src/compare_zero_shot_vs_trained.py`) evaluates both methods on the same 10 synthetic test images and generates a side-by-side bar chart, making the performance difference immediately visible.

### Educational Message

> A zero-shot segmentation baseline can produce useful masks without training, but a model trained on task-specific synthetic data (100 images) can significantly improve instance-level segmentation performance.

### Note on Real Data / 実データに関する注意

In real research, models trained on synthetic data may perform differently on experimental images. This gap is called **sim2real gap**. This public demo does not include real experimental data — it uses synthetic data to demonstrate the evaluation workflow.

## Quick Start / はじめかた

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above and run cells from top to bottom.

### Option 2: Local Execution

```bash
git clone https://github.com/fanfanfuzzy/nanosheet-segmentation-colab-demo.git
cd nanosheet-segmentation-colab-demo
pip install -r requirements.txt

# Generate synthetic data (for visualization)
python src/generate_synthetic_nanosheets.py \
    --config configs/synthetic_mid.yaml \
    --num-images 10 \
    --output-dir outputs/synthetic_demo

# Visualize dataset
python src/visualize_dataset.py \
    --input-dir outputs/synthetic_demo \
    --output-dir outputs/figures

# Compare zero-shot vs YOLO-seg on shared test images
python src/compare_zero_shot_vs_trained.py \
    --gt-dir demo_assets/ground_truth \
    --zero-shot-dir demo_assets/predictions_sam_baseline \
    --trained-dir demo_assets/predictions_yolo_trained \
    --output-csv outputs/comparison_metrics.csv \
    --output-fig outputs/comparison_barplot.png
```

## Workshop Structure / ワークショップ構成

| Step | Content |
|------|---------|
| 1–2 | Clone repository & install dependencies |
| 3 | Generate synthetic nanosheet images |
| 4 | Visualize dataset and ground-truth masks |
| 5 | Load SAM (ViT-H) AMG predictions on test images |
| 6 | Load pre-computed YOLO-seg predictions |
| 7 | Evaluate both methods with instance-level metrics |
| 8 | Compare metrics with bar chart |
| 9 | Per-image visual comparison (GT vs SAM vs YOLO-seg) |
| 10 | (Optional) Short YOLO training demo |

## YOLO-seg Training Details

The included model (`models/yolo11s-seg-nanosheet.pt`) was trained as follows:

- **Base model:** YOLOv11s-seg (Ultralytics)
- **Training data:** 100 synthetic nanosheet images (512×512, Beer-Lambert model)
- **Validation data:** 10 synthetic test images
- **Training:** 150 epochs max, early stopping (patience=30), stopped at epoch 83 (best epoch 53)
- **Best validation Mask mAP50:** 0.983
- **GPU:** NVIDIA RTX A6000

Training and inference scripts are available in [2603-nanosheet-overlap-segmentation](https://github.com/fanfanfuzzy/2603-nanosheet-overlap-segmentation) (branch: `devin/colab-demo-yolo-pipeline`).

## Evaluation Metrics / 評価指標

| Metric | Description |
|--------|-------------|
| `instance_recall` | 正解ナノシートのうちモデルが見つけた割合 |
| `instance_precision` | モデルの予測のうち正解だった割合 |
| `instance_f1` | RecallとPrecisionの調和平均 |
| `mean_best_iou` | 各正解に対する最良IoUの平均 |
| `semantic_iou` | 前景領域全体のピクセルレベルIoU |

## Project Structure

```
nanosheet-segmentation-colab-demo/
├── README.md
├── LICENSE                          (MIT)
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── nanosheet_segmentation_demo.ipynb
├── src/
│   ├── generate_synthetic_nanosheets.py
│   ├── visualize_dataset.py
│   ├── sam_zero_shot_baseline.py
│   ├── evaluate_predictions.py
│   ├── compare_metrics.py
│   ├── compare_zero_shot_vs_trained.py
│   └── utils/
│       ├── masks.py
│       ├── metrics.py
│       └── plotting.py
├── models/
│   └── yolo11s-seg-nanosheet.pt     (YOLO-seg trained on synthetic data)
├── configs/
│   ├── synthetic_easy.yaml
│   ├── synthetic_mid.yaml
│   └── synthetic_hard.yaml
├── demo_assets/
│   ├── README.md
│   ├── test_images/                 (10 synthetic test images)
│   ├── ground_truth/                (GT label maps for test images)
│   ├── predictions_sam_baseline/    (zero-shot baseline results)
│   └── predictions_yolo_trained/    (YOLO-seg inference results)
├── outputs/                         (git-ignored except .gitkeep)
└── docs/
    ├── workshop_plan.md
    ├── data_policy.md
    └── method_overview.md
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Note: YOLO-seg (Ultralytics) is AGPL-3.0 licensed. The model weights included here were produced using Ultralytics and are subject to their license terms.
