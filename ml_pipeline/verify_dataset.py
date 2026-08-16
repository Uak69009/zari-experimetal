"""Comprehensive Validation Script for ZARI.ai Final Dataset.

Verifies:
1. Row Count (123,300)
2. Split Counts (train=110008, val=6644, test=6648)
3. Class Counts (67 head classes, 39 pretrain classes)
4. SHA256 Hash Leakage Across Splits (0 leakage)
5. Field-Only Val/Test (val and test contain ONLY plantcity and nwrd)
6. Class Mapping (class_map_final.json structure and counts)
7. File Path Sampling (1,000 random samples path resolution check)

Outputs report to ml_pipeline/ANALYSIS_COMPLETE/reports/final_verification.txt
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
INPUT_CSV = DATA_DIR / "dataset_final_training.csv"
CLASS_MAP_JSON = DATA_DIR / "class_map_final.json"
REPORT_TXT = SCRIPT_DIR / "ANALYSIS_COMPLETE" / "reports" / "final_verification.txt"
RAW_ROOT = DATA_DIR / "raw"


def resolve_path(image_path_str: str) -> Path:
    candidate = Path(image_path_str)
    if candidate.exists():
        return candidate

    match = re.search(r"raw[\\/](.+)$", str(image_path_str), flags=re.IGNORECASE)
    if match:
        suffix = match.group(1).replace("\\", "/")
        candidate = RAW_ROOT / Path(suffix)
        if candidate.exists():
            return candidate

    return candidate


def main() -> None:
    print("=" * 65)
    print("  ZARI.ai — FINAL DATASET VERIFICATION GATEWAY")
    print("=" * 65)

    if not INPUT_CSV.exists():
        print(f"❌ FAIL: Missing input CSV at {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    results: list[tuple[str, bool, str]] = []
    report_lines: list[str] = [
        "ZARI.ai Final Dataset Verification Report",
        "=========================================",
        f"Input CSV: {INPUT_CSV}",
        f"Class Map: {CLASS_MAP_JSON}",
        "",
    ]

    # CHECK 1: ROW COUNT
    try:
        total = len(df)
        assert total == 123300, f"Expected 123300 rows, got {total}"
        msg = f"Total rows: {total:,} == 123,300"
        results.append(("1. ROW COUNT", True, msg))
    except AssertionError as e:
        results.append(("1. ROW COUNT", False, str(e)))

    # CHECK 2: SPLIT COUNTS
    try:
        split_counts = df["split"].value_counts()
        train_c = split_counts.get("train", 0)
        val_c = split_counts.get("val", 0)
        test_c = split_counts.get("test", 0)

        assert train_c == 110008, f"Expected train=110008, got {train_c}"
        assert val_c == 6644, f"Expected val=6644, got {val_c}"
        assert test_c == 6648, f"Expected test=6648, got {test_c}"
        msg = f"train={train_c:,}, val={val_c:,}, test={test_c:,}"
        results.append(("2. SPLIT COUNTS", True, msg))
    except AssertionError as e:
        results.append(("2. SPLIT COUNTS", False, str(e)))

    # CHECK 3: CLASS COUNTS
    try:
        head_classes = df[df["class_id"] >= 0]["class_id"].nunique()
        pretrain_classes = df[df["pretrain_id"] >= 0]["pretrain_id"].nunique()

        assert head_classes == 67, f"Expected 67 head classes, got {head_classes}"
        assert pretrain_classes == 39, f"Expected 39 pretrain classes, got {pretrain_classes}"
        msg = f"Head classes: {head_classes}, Pretrain classes: {pretrain_classes}"
        results.append(("3. CLASS COUNTS", True, msg))
    except AssertionError as e:
        results.append(("3. CLASS COUNTS", False, str(e)))

    # CHECK 4: HASH LEAKAGE CHECK
    try:
        hash_splits = df.groupby("sha256")["split"].nunique()
        leakage = int((hash_splits > 1).sum())
        assert leakage == 0, f"Found {leakage} hashes spanning multiple splits"
        msg = f"0 hashes cross split boundaries ({df['sha256'].nunique():,} unique hashes)"
        results.append(("4. HASH LEAKAGE CHECK", True, msg))
    except AssertionError as e:
        results.append(("4. HASH LEAKAGE CHECK", False, str(e)))

    # CHECK 5: FIELD-ONLY VAL/TEST
    try:
        val_test_sources = set(df[df["split"].isin(["val", "test"])]["source_dataset"].unique())
        assert val_test_sources <= {"plantcity", "nwrd"}, f"Lab data in val/test: {val_test_sources}"
        msg = f"Val/Test sources: {sorted(list(val_test_sources))} (100% field images)"
        results.append(("5. FIELD-ONLY VAL/TEST", True, msg))
    except AssertionError as e:
        results.append(("5. FIELD-ONLY VAL/TEST", False, str(e)))

    # CHECK 6: CLASS MAPPING
    try:
        if not CLASS_MAP_JSON.exists():
            raise AssertionError(f"Missing {CLASS_MAP_JSON}")
        with open(CLASS_MAP_JSON, "r", encoding="utf-8") as f:
            class_map = json.load(f)

        json_head = len(class_map.get("head_classes", {}))
        json_pretrain = len(class_map.get("pretrain_classes", {}))
        has_merge = ("label_merge_map" in class_map) or ("merge_map" in class_map)

        assert json_head == 67, f"Expected 67 head classes in JSON, got {json_head}"
        assert json_pretrain == 39, f"Expected 39 pretrain classes in JSON, got {json_pretrain}"
        assert has_merge, "Missing merge map in class_map_final.json"
        msg = f"JSON verified: head_classes={json_head}, pretrain_classes={json_pretrain}, merge_map=Present"
        results.append(("6. CLASS MAPPING", True, msg))
    except Exception as e:
        results.append(("6. CLASS MAPPING", False, str(e)))

    # CHECK 7: FILE PATH SAMPLING
    try:
        sample_df = df.sample(n=min(1000, len(df)), random_state=42)
        valid_count = 0
        for p in sample_df["image_path"].astype(str):
            if resolve_path(p).exists():
                valid_count += 1
        hit_rate = (valid_count / len(sample_df)) * 100
        assert hit_rate == 100.0, f"Sampled path hit rate {hit_rate:.2f}% < 100%"
        msg = f"Sampled 1,000 images: {valid_count}/1,000 exist ({hit_rate:.2f}% hit rate)"
        results.append(("7. FILE PATH SAMPLING", True, msg))
    except Exception as e:
        results.append(("7. FILE PATH SAMPLING", False, str(e)))

    # PRINT AND LOG RESULTS
    all_passed = True
    print("\nCheck Results:")
    print("-" * 65)

    for name, passed, detail in results:
        status_icon = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"{status_icon} | {name.ljust(22)} | {detail}")
        report_lines.append(f"{status_icon} | {name.ljust(22)} | {detail}")

    report_lines.append("")
    report_lines.append(f"Overall Verification Outcome: {'PASSED' if all_passed else 'FAILED'}")

    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    print("-" * 65)
    print(f"Detailed verification report saved to: {REPORT_TXT}")

    if not all_passed:
        print("\n❌ CRITICAL: Verification FAILED. Stopping execution.")
        sys.exit(1)
    else:
        print("\n✅ SUCCESS: All verification checks PASSED. Dataset is 100% training-ready!")


if __name__ == "__main__":
    main()
