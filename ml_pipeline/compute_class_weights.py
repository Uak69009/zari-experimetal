"""Class Imbalance Analysis and Weight Computation for ZARI.ai.

This script:
1. Loads dataset_final_training.csv.
2. Analyzes class distribution across all 106 pretrain classes and 67 head classes.
3. Computes inverse-frequency class weights for both pretrain and head stages.
4. Normalizes and clips weights to prevent gradient instability during backpropagation.
5. Saves class_weights.json and generates a comprehensive analysis report.

WHY CLASS WEIGHTS ARE APPLIED ONLY TO TRAINING:
- Training loss uses class weights to ensure minority classes contribute equally to parameter updates.
- Validation and test loss/metrics MUST remain UNWEIGHTED to give an honest, unskewed evaluation
  reflecting true real-world class distributions and field performance.

WEIGHT FORMULA:
  weight_i = total_samples / (num_classes * count_i)
  - Classes with fewer samples get weight > 1.0.
  - Classes with more samples get weight < 1.0.

WHY CLIPPING IS NECESSARY:
  Extremely small classes (e.g. count=60) can yield raw weights > 15x.
  Without clipping (min=0.1, max=10.0), a single wrong prediction on a rare class
  can trigger massive gradient spikes and destabilize training.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INPUT_CSV = DATA_DIR / "dataset_final_training.csv"
OUTPUT_WEIGHTS_JSON = DATA_DIR / "class_weights.json"
REPORT_TXT = SCRIPT_DIR / "ANALYSIS_COMPLETE" / "reports" / "class_weights_report.txt"


def compute_inverse_frequency_weights(
    counts: dict[str | int, int], total_samples: int, min_clip: float = 0.1, max_clip: float = 10.0
) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    """Compute normalized and clipped inverse-frequency class weights.

    Formula:
      1. raw_weight_i = total_samples / (num_classes * count_i)
      2. norm_weight_i = raw_weight_i / mean(raw_weights)
      3. clipped_weight_i = clamp(norm_weight_i, min_clip, max_clip)
    """
    keys = list(counts.keys())
    count_arr = np.array([counts[k] for k in keys], dtype=np.float64)
    num_classes = len(keys)

    # 1. Raw inverse-frequency formula
    raw_weights = total_samples / (num_classes * count_arr)

    # 2. Normalize so mean weight == 1.0
    norm_weights = raw_weights / np.mean(raw_weights)

    # 3. Clip weights to prevent gradient explosion
    clipped_weights = np.clip(norm_weights, min_clip, max_clip)

    weights_dict = {str(k): float(np.round(w, 4)) for k, w in zip(keys, clipped_weights)}
    raw_weights_dict = {str(k): float(np.round(w, 4)) for k, w in zip(keys, norm_weights)}

    return weights_dict, raw_weights_dict, clipped_weights


def main() -> None:
    print("=" * 65)
    print("  ZARI.ai — CLASS IMBALANCE & WEIGHT COMPUTATION PIPELINE")
    print("=" * 65)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing dataset CSV at {INPUT_CSV}")

    # 1. Load Dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"\n[STEP 1] Loaded dataset: {INPUT_CSV}")
    print(f"Total Rows: {len(df):,}, Total Columns: {len(df.columns)}")

    train_df = df[df["split"] == "train"].copy()
    field_train_df = train_df[train_df["source_dataset"].isin(["plantcity", "nwrd"])].copy()

    # 2. Analyze Pretrain Classes (All 106 Classes in train split)
    print("\n[STEP 2] Analyzing Pretrain Classes (All 106 Classes)...")
    pretrain_counts_series = train_df["class_name"].value_counts().sort_index()
    pretrain_counts = pretrain_counts_series.to_dict()

    all_sorted_class_names = sorted(list(pretrain_counts.keys()))
    pretrain_name_to_idx = {name: idx for idx, name in enumerate(all_sorted_class_names)}
    pretrain_idx_counts = {pretrain_name_to_idx[k]: pretrain_counts[k] for k in all_sorted_class_names}

    p_total = len(train_df)
    p_num_classes = len(pretrain_counts)
    p_min = int(min(pretrain_counts.values()))
    p_max = int(max(pretrain_counts.values()))
    p_mean = float(np.mean(list(pretrain_counts.values())))
    p_median = float(np.median(list(pretrain_counts.values())))
    p_imbalance_ratio = float(p_max / p_min)

    p_under_30 = [k for k, v in pretrain_counts.items() if v < 30]
    p_over_5000 = [k for k, v in pretrain_counts.items() if v > 5000]

    print(f"✓ Total Train Samples: {p_total:,}")
    print(f"✓ Total Classes: {p_num_classes}")
    print(f"✓ Min Count: {p_min} | Max Count: {p_max:,}")
    print(f"✓ Mean Count: {p_mean:.1f} | Median Count: {p_median:.1f}")
    print(f"✓ Imbalance Ratio (Max/Min): {p_imbalance_ratio:.2f}x")
    print(f"✓ Classes with < 30 samples : {len(p_under_30)} {p_under_30}")
    print(f"✓ Classes with > 5000 samples: {len(p_over_5000)} {p_over_5000}")

    # 3. Analyze Head Classes (67 Field Classes in PlantCity + NWRD train)
    print("\n[STEP 3] Analyzing Head Classes (67 Field Classes)...")
    head_counts_series = field_train_df.groupby("class_id")["class_name"].count().sort_index()
    head_counts = {int(k): int(v) for k, v in head_counts_series.to_dict().items()}

    head_name_map = field_train_df.groupby("class_id")["class_name"].first().to_dict()

    h_total = len(field_train_df)
    h_num_classes = len(head_counts)
    h_min = int(min(head_counts.values()))
    h_max = int(max(head_counts.values()))
    h_mean = float(np.mean(list(head_counts.values())))
    h_median = float(np.median(list(head_counts.values())))
    h_imbalance_ratio = float(h_max / h_min)

    h_under_30 = [f"{k}:{head_name_map[k]}" for k, v in head_counts.items() if v < 30]
    h_over_5000 = [f"{k}:{head_name_map[k]}" for k, v in head_counts.items() if v > 5000]

    print(f"✓ Total Field Train Samples: {h_total:,}")
    print(f"✓ Total Head Classes: {h_num_classes}")
    print(f"✓ Min Count: {h_min} | Max Count: {h_max:,}")
    print(f"✓ Mean Count: {h_mean:.1f} | Median Count: {h_median:.1f}")
    print(f"✓ Imbalance Ratio (Max/Min): {h_imbalance_ratio:.2f}x")
    print(f"✓ Classes with < 30 samples : {len(h_under_30)}")
    print(f"✓ Classes with > 5000 samples: {len(h_over_5000)}")

    # 4. Compute Weights
    print("\n[STEP 4] Computing Inverse-Frequency Class Weights...")
    pretrain_weights_by_idx, pretrain_raw_dict, pretrain_clip_arr = compute_inverse_frequency_weights(
        pretrain_idx_counts, p_total
    )

    pretrain_weights_by_name = {
        name: pretrain_weights_by_idx[str(pretrain_name_to_idx[name])] for name in all_sorted_class_names
    }

    head_weights_by_id, head_raw_dict, head_clip_arr = compute_inverse_frequency_weights(head_counts, h_total)

    head_weights_by_name = {head_name_map[int(k)]: head_weights_by_id[k] for k in head_weights_by_id}

    # Identify extreme weights
    extreme_pretrain_high = {k: v for k, v in pretrain_weights_by_name.items() if v > 5.0}
    extreme_pretrain_low = {k: v for k, v in pretrain_weights_by_name.items() if v < 0.2}

    extreme_head_high = {k: v for k, v in head_weights_by_name.items() if v > 5.0}
    extreme_head_low = {k: v for k, v in head_weights_by_name.items() if v < 0.2}

    # Loss recommendation
    if p_imbalance_ratio > 500.0 or h_imbalance_ratio > 500.0:
        loss_recommendation = "RECOMMEND FOCAL LOSS (Extreme Imbalance > 500:1 detected)"
    else:
        loss_recommendation = (
            "RECOMMEND WEIGHTED CROSS ENTROPY LOSS (Imbalance ratio <= 500:1, standard weighted CE is optimal)"
        )

    # 5. Save Weights JSON
    print("\n[STEP 5] Saving Class Weights JSON...")
    weights_json_payload = {
        "pretrain_weights": pretrain_weights_by_idx,
        "head_weights": head_weights_by_id,
        "pretrain_weights_by_name": pretrain_weights_by_name,
        "head_weights_by_name": head_weights_by_name,
        "pretrain_stats": {
            "total_samples": p_total,
            "num_classes": p_num_classes,
            "imbalance_ratio": round(p_imbalance_ratio, 2),
            "min_count": p_min,
            "max_count": p_max,
            "mean_count": round(p_mean, 2),
            "median_count": round(p_median, 2),
        },
        "head_stats": {
            "total_samples": h_total,
            "num_classes": h_num_classes,
            "imbalance_ratio": round(h_imbalance_ratio, 2),
            "min_count": h_min,
            "max_count": h_max,
            "mean_count": round(h_mean, 2),
            "median_count": round(h_median, 2),
        },
        "recommendation": loss_recommendation,
    }

    OUTPUT_WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_WEIGHTS_JSON.write_text(json.dumps(weights_json_payload, indent=2), encoding="utf-8")
    print(f"✓ Saved JSON: {OUTPUT_WEIGHTS_JSON}")

    # 6. Save Report
    report_lines: list[str] = []
    report_lines.append("ZARI.ai Class Imbalance & Weight Computation Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Input CSV: {INPUT_CSV}")
    report_lines.append(f"Output JSON: {OUTPUT_WEIGHTS_JSON}")
    report_lines.append("")
    report_lines.append("1. Pretrain Phase (106 Classes)")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Train Samples: {p_total:,}")
    report_lines.append(f"Total Classes: {p_num_classes}")
    report_lines.append(f"Min Class Count: {p_min} | Max Class Count: {p_max:,}")
    report_lines.append(f"Mean Count: {p_mean:.2f} | Median Count: {p_median:.2f}")
    report_lines.append(f"Imbalance Ratio: {p_imbalance_ratio:.2f}x")
    report_lines.append(f"Clipped Weight Range: [{min(pretrain_clip_arr):.4f}, {max(pretrain_clip_arr):.4f}]")
    report_lines.append(f"Classes with < 30 samples: {p_under_30}")
    report_lines.append(f"Classes with > 5000 samples: {p_over_5000}")
    report_lines.append(f"High Extreme Weights (> 5.0): {extreme_pretrain_high}")
    report_lines.append(f"Low Extreme Weights (< 0.2): {extreme_pretrain_low}")
    report_lines.append("")
    report_lines.append("2. Head Fine-Tuning Phase (67 Field Classes)")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Field Train Samples: {h_total:,}")
    report_lines.append(f"Total Head Classes: {h_num_classes}")
    report_lines.append(f"Min Class Count: {h_min} | Max Class Count: {h_max:,}")
    report_lines.append(f"Mean Count: {h_mean:.2f} | Median Count: {h_median:.2f}")
    report_lines.append(f"Imbalance Ratio: {h_imbalance_ratio:.2f}x")
    report_lines.append(f"Clipped Weight Range: [{min(head_clip_arr):.4f}, {max(head_clip_arr):.4f}]")
    report_lines.append(f"Classes with < 30 samples: {h_under_30}")
    report_lines.append(f"Classes with > 5000 samples: {h_over_5000}")
    report_lines.append(f"High Extreme Weights (> 5.0): {extreme_head_high}")
    report_lines.append(f"Low Extreme Weights (< 0.2): {extreme_head_low}")
    report_lines.append("")
    report_lines.append("3. Loss Function Recommendation")
    report_lines.append("-" * 40)
    report_lines.append(f"Outcome: {loss_recommendation}")
    report_lines.append("")
    report_lines.append("Pretrain Full Class Distribution Table")
    report_lines.append("-" * 60)
    for idx, name in enumerate(all_sorted_class_names):
        cnt = pretrain_counts[name]
        w = pretrain_weights_by_name[name]
        report_lines.append(f"[{str(idx).rjust(3)}] {name.ljust(45)} | count: {str(cnt).rjust(5)} | weight: {w:.4f}")

    report_lines.append("")
    report_lines.append("Head Full Class Distribution Table")
    report_lines.append("-" * 60)
    for cid in sorted(list(head_counts.keys())):
        cnt = head_counts[cid]
        cname = head_name_map[cid]
        w = head_weights_by_id[str(cid)]
        report_lines.append(f"[{str(cid).rjust(2)}] {cname.ljust(45)} | count: {str(cnt).rjust(5)} | weight: {w:.4f}")

    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"✓ Saved Report: {REPORT_TXT}")

    # 7. Print Console Summary
    print("\n" + "=" * 65)
    print("  SUMMARY OUTPUT")
    print("=" * 65)
    print(f"Pretrain Weight Range : [{min(pretrain_clip_arr):.4f}, {max(pretrain_clip_arr):.4f}]")
    print(f"Head Weight Range     : [{min(head_clip_arr):.4f}, {max(head_clip_arr):.4f}]")
    print(f"Extreme Pretrain (>5) : {extreme_pretrain_high}")
    print(f"Extreme Pretrain (<0.2): {extreme_pretrain_low}")
    print(f"Extreme Head (>5)     : {extreme_head_high}")
    print(f"Extreme Head (<0.2)   : {extreme_head_low}")
    print(f"Loss Recommendation   : {loss_recommendation}")


if __name__ == "__main__":
    main()
