# 90-Minute Workshop Plan / 90分ワークショップ計画

## Overview / 概要

This workshop introduces instance segmentation using synthetic nanosheet-like images.
Participants will compare a zero-shot baseline with a trained model and learn
how to evaluate segmentation performance.

## Time Allocation / 時間配分

| Time | Section | Content |
|------|---------|---------|
| 0-5 min | Introduction | Workshop goals, data policy explanation |
| 5-15 min | Setup | Clone repo, install dependencies in Colab |
| 15-25 min | Data Generation | Generate synthetic nanosheet images |
| 25-35 min | Visualization | View images and ground-truth masks |
| 35-45 min | Zero-shot Baseline | Run and view baseline predictions |
| 45-55 min | Trained Model | Load precomputed trained-model results |
| 55-70 min | Evaluation | Compute and compare metrics |
| 70-80 min | Discussion | sim2real gap, metric interpretation |
| 80-90 min | Wrap-up & Demo | Show Devin PR, Q&A |

## Key Messages / キーメッセージ

1. **Zero-shot methods** are useful because they require no training data.
2. **Trained models** can improve performance when task-specific labels are available.
3. **Evaluation metrics** help quantify improvement objectively.
4. **sim2real gap** means synthetic-trained models may perform differently on real data.
5. **AI-assisted coding** (e.g., Devin) can help write and update analysis pipelines,
   but human review of diffs is essential.

## Prerequisites / 前提条件

- Google account (for Colab)
- No GPU required
- No programming experience required (cells run top-to-bottom)
- No GitHub account required for participants

## Optional Extension / 発展（任意）

- Short YOLO training demo (requires GPU runtime)
- Adding new evaluation metrics
- Trying different difficulty levels
