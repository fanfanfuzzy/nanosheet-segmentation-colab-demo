# Nanosheet Segmentation Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fanfanfuzzy/nanosheet-segmentation-colab-demo/blob/main/notebooks/nanosheet_segmentation_demo.ipynb)

**Beginner-friendly Google Colab demo for synthetic nanosheet instance segmentation.**

This workshop demonstrates the full workflow from synthetic data generation to segmentation prediction, evaluation, and visualization — comparing SAM (zero-shot) with YOLO-seg (trained) across three difficulty levels (easy / mid / hard).

## Data Policy / データ公開方針

> **This public repository does not contain real experimental TEM images, private annotations, or unpublished experimental datasets.**
>
> All images, masks, model weights, predictions, and sample metrics included in this repository are synthetic or generated for educational purposes.
> The YOLO-seg model weights (`models/yolo11s-seg-nanosheet-{easy,mid,hard}.pt`) were trained exclusively on synthetic nanosheet images.

> **このリポジトリには、実際の実験TEM画像、手動アノテーション、未公開の実験データは含めていません。**
>
> 含まれる画像・マスク・モデルweights・予測結果は、教材用に生成した人工ナノシート画像、またはそれに基づくサンプル結果です。
> YOLO-segモデルは合成画像のみで学習しています。

## What This Demo Shows / このデモで学ぶこと

This demo compares three approaches to instance segmentation across three difficulty levels:

1. **SAM (ViT-H) zero-shot** — Segment Anything Model with Automatic Mask Generator (AMG). No task-specific training. A foundation model that works on any image domain.

2. **YOLO-seg pretrained** — YOLOv11s-seg with COCO pretrained weights only. Shows that general-purpose object detection models cannot segment nanosheets without fine-tuning.

3. **YOLO-seg trained** — YOLOv11s-seg fine-tuned on 100 synthetic nanosheet images per difficulty level. Demonstrates that task-specific training dramatically improves performance.

### Difficulty Levels

Synthetic data is generated using the Beer-Lambert attenuation model with three difficulty presets from [2603-nanosheet-overlap-segmentation](https://github.com/fanfanfuzzy/2603-nanosheet-overlap-segmentation):

| Difficulty | SNR | Description |
|-----------|-----|-------------|
| Easy | ~2.7 | High contrast, low noise |
| Mid | ~1.7 | Moderate contrast and noise |
| Hard | ~1.1 | Low contrast, high noise |

### Educational Message

> As difficulty increases, SAM performance degrades significantly while YOLO-seg trained on task-specific data maintains better performance — demonstrating the value of domain-specific training data.

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

# Visualize per-difficulty comparison
python src/visualize_comparison.py \
    --image-dir demo_assets/mid/test_images \
    --gt-dir demo_assets/mid/ground_truth \
    --zero-shot-dir demo_assets/mid/predictions_sam \
    --pretrained-dir demo_assets/mid/predictions_yolo_pretrained \
    --trained-dir demo_assets/mid/predictions_yolo_trained \
    --output-dir outputs/comparison_visual_mid
```

## Workshop Structure / ワークショップ構成

| Step | Content |
|------|---------|
| 1–2 | Clone repository & install dependencies |
| 3 | Generate synthetic nanosheet images |
| 4 | Visualize dataset and ground-truth masks |
| 5 | Load SAM (ViT-H) AMG predictions for each difficulty |
| 6 | Load YOLO-seg pretrained & trained predictions |
| 7 | Evaluate all methods with instance-level metrics |
| 8 | Compare metrics across difficulty levels |
| 9 | Visualize per-image comparison grids for each difficulty |
| 10 | (Optional) Short YOLO training demo |

## YOLO-seg Training Details

Three separate models were trained, one per difficulty level:

| Difficulty | Model File | Best Mask mAP50 |
|-----------|-----------|-----------------|
| Easy | `models/yolo11s-seg-nanosheet-easy.pt` | 0.845 |
| Mid | `models/yolo11s-seg-nanosheet-mid.pt` | 0.722 |
| Hard | `models/yolo11s-seg-nanosheet-hard.pt` | 0.479 |

- **Base model:** YOLOv11s-seg (Ultralytics)
- **Training data:** 100 synthetic nanosheet images per difficulty (512×512, Beer-Lambert model)
- **Validation data:** 10 synthetic test images per difficulty
- **Training:** 150 epochs, early stopping (patience=30)
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
│   ├── visualize_comparison.py
│   ├── sam_zero_shot_baseline.py
│   ├── evaluate_predictions.py
│   ├── compare_metrics.py
│   ├── compare_zero_shot_vs_trained.py
│   └── utils/
│       ├── masks.py
│       ├── metrics.py
│       └── plotting.py
├── models/
│   ├── yolo11s-seg-nanosheet-easy.pt
│   ├── yolo11s-seg-nanosheet-mid.pt
│   └── yolo11s-seg-nanosheet-hard.pt
├── configs/
│   ├── synthetic_easy.yaml
│   ├── synthetic_mid.yaml
│   └── synthetic_hard.yaml
├── demo_assets/
│   ├── README.md
│   ├── easy/                        (test images, GT, predictions)
│   ├── mid/                         (test images, GT, predictions)
│   └── hard/                        (test images, GT, predictions)
├── outputs/                         (git-ignored except .gitkeep)
└── docs/
    ├── workshop_plan.md
    ├── data_policy.md
    └── method_overview.md
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Note: YOLO-seg (Ultralytics) is AGPL-3.0 licensed. The model weights included here were produced using Ultralytics and are subject to their license terms.
