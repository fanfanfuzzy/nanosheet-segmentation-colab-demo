# Demo Assets / デモ用アセット

This directory contains pre-generated synthetic data for the Colab demo.

**All data in this directory is synthetic.** No real experimental images are included.

## Contents

- `synthetic_images/` — Pre-generated synthetic nanosheet images (PNG)
- `synthetic_gt/` — Ground-truth instance masks for the synthetic images
- `predictions_sam_baseline/` — Zero-shot baseline predictions
- `predictions_yolo_trained/` — Simulated trained-model predictions
- `sample_metrics/` — Pre-computed evaluation metrics (CSV)

## Purpose

These assets allow the Colab notebook to demonstrate the full evaluation
pipeline without requiring participants to run long computations.

Participants can also regenerate these assets themselves using the
scripts in `src/`.
