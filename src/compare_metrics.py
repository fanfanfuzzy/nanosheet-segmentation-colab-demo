"""Compare metrics from multiple methods and generate visualization.

Usage:
    python src/compare_metrics.py \
        --inputs outputs/metrics_sam.csv outputs/metrics_yolo.csv \
        --output-csv outputs/comparison_metrics.csv \
        --output-fig outputs/comparison_barplot.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.plotting import plot_metric_comparison


METRIC_COLUMNS = [
    "instance_recall",
    "instance_precision",
    "instance_f1",
    "mean_best_iou",
    "semantic_iou",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics from multiple segmentation methods."
    )
    parser.add_argument("--inputs", type=str, nargs="+", required=True,
                        help="Input CSV files with metrics.")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Output CSV for comparison table.")
    parser.add_argument("--output-fig", type=str, required=True,
                        help="Output path for bar chart figure.")
    args = parser.parse_args()

    # Load all metric files
    all_summaries = []
    metrics_dict = {}

    for csv_path in args.inputs:
        path = Path(csv_path)
        if not path.exists():
            print(f"Warning: File not found, skipping: {csv_path}")
            continue

        df = pd.read_csv(path)

        # Get the AVERAGE row
        avg_row = df[df["image"] == "AVERAGE"]
        if avg_row.empty:
            # Compute average ourselves
            numeric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
            avg_values = df[numeric_cols].mean()
            method_name = df["method"].iloc[0] if "method" in df.columns else path.stem
        else:
            avg_values = avg_row.iloc[0]
            method_name = avg_values.get("method", path.stem)

        method_metrics = {}
        for col in METRIC_COLUMNS:
            if col in avg_values.index if hasattr(avg_values, "index") else col in avg_values:
                method_metrics[col] = float(avg_values[col])

        metrics_dict[method_name] = method_metrics
        all_summaries.append({"method": method_name, **method_metrics})

    if not all_summaries:
        print("Error: No valid metric files found.")
        sys.exit(1)

    # Save comparison CSV
    comparison_df = pd.DataFrame(all_summaries)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv, index=False)
    print(f"Comparison CSV saved to: {output_csv}")

    # Generate bar chart
    plot_metric_comparison(metrics_dict, args.output_fig)
    print(f"Comparison figure saved to: {args.output_fig}")

    # Print table
    print("\n--- Metric Comparison ---")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
