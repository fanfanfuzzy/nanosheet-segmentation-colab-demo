# Nanosheet Segmentation Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fanfanfuzzy/nanosheet-segmentation-colab-demo/blob/main/notebooks/nanosheet_segmentation_demo.ipynb)

**Beginner-friendly Google Colab demo for synthetic nanosheet instance segmentation.**

This workshop demonstrates the full workflow from synthetic data generation to segmentation prediction, evaluation, and visualization — comparing a zero-shot baseline with a trained segmentation model.

## Data Policy / データ公開方針

> **This public repository does not contain real experimental TEM images, private annotations, or unpublished experimental datasets.**
>
> All images, masks, predictions, and sample metrics included in this repository are synthetic or generated for educational purposes.

> **このリポジトリには、実際の実験TEM画像、手動アノテーション、未公開の実験データは含めていません。**
>
> 含まれる画像・マスク・予測結果は、教材用に生成した人工ナノシート画像、またはそれに基づくサンプル結果です。

## What This Demo Shows / このデモで学ぶこと

This demo compares two approaches to instance segmentation:

1. **Zero-shot segmentation baseline** — Uses image processing heuristics (adaptive thresholding, morphological operations, connected components) to produce candidate masks **without any training**. This represents approaches inspired by SAM-style mask proposals.

2. **Trained segmentation model** — Uses predictions from a model trained on synthetic nanosheet data. The key message: *a model trained on task-specific synthetic data can improve instance-level segmentation performance compared to a training-free baseline.*

### Educational Message

> A zero-shot segmentation baseline can produce useful masks without training, but a model trained on task-specific synthetic data can improve instance-level segmentation performance.

## Quick Start / はじめかた

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above and run cells from top to bottom.

### Option 2: Local Execution

```bash
git clone https://github.com/fanfanfuzzy/nanosheet-segmentation-colab-demo.git
cd nanosheet-segmentation-colab-demo
pip install -r requirements.txt

# Generate synthetic data
python src/generate_synthetic_nanosheets.py \
    --config configs/synthetic_mid.yaml \
    --num-images 10 \
    --output-dir outputs/synthetic_demo

# Visualize dataset
python src/visualize_dataset.py \
    --input-dir outputs/synthetic_demo \
    --output-dir outputs/figures

# Run zero-shot baseline
python src/sam_zero_shot_baseline.py \
    --input-dir outputs/synthetic_demo \
    --output-dir outputs/predictions_sam_baseline

# Evaluate
python src/evaluate_predictions.py \
    --gt-dir outputs/synthetic_demo/masks \
    --pred-dir outputs/predictions_sam_baseline \
    --method-name sam_zero_shot \
    --output outputs/metrics_sam.csv
```

## Workshop Structure / ワークショップ構成

| Step | Content |
|------|---------|
| 1 | Clone repository & install dependencies |
| 2 | Generate synthetic nanosheet images |
| 3 | Visualize dataset and ground-truth masks |
| 4 | Run zero-shot segmentation baseline |
| 5 | Load trained-model predictions |
| 6 | Evaluate both methods |
| 7 | Compare metrics with bar chart |
| 8 | (Optional) Short YOLO training demo |

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
│   └── utils/
│       ├── masks.py
│       ├── metrics.py
│       └── plotting.py
├── configs/
│   ├── synthetic_easy.yaml
│   ├── synthetic_mid.yaml
│   └── synthetic_hard.yaml
├── demo_assets/
│   ├── README.md
│   ├── synthetic_images/
│   ├── synthetic_gt/
│   ├── predictions_sam_baseline/
│   ├── predictions_yolo_trained/
│   └── sample_metrics/
├── outputs/                         (git-ignored except .gitkeep)
└── docs/
    ├── workshop_plan.md
    ├── data_policy.md
    └── method_overview.md
```

## License

Code: MIT License

This repository does not grant any license to real experimental data because no real experimental data are included.
