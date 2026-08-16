import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ml_pipeline" / "data"

V3_CSV_PATH = DATA_DIR / "dataset_final_training_v3.csv"

def main():
    print("=====================================================================")
    print("  ZARI.ai — Master Dataset V3 Cross-Split Leakage Audit")
    print("=====================================================================\n")

    if not V3_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing V3 CSV at {V3_CSV_PATH}")

    df = pd.read_csv(V3_CSV_PATH, low_memory=False)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_hashes = set(train_df["sha256"].dropna())
    val_hashes = set(val_df["sha256"].dropna())
    test_hashes = set(test_df["sha256"].dropna())

    tv_leak = train_hashes.intersection(val_hashes)
    tt_leak = train_hashes.intersection(test_hashes)
    vt_leak = val_hashes.intersection(test_hashes)

    print(f"Train Set Hashes : {len(train_hashes):,}")
    print(f"Val Set Hashes   : {len(val_hashes):,}")
    print(f"Test Set Hashes  : {len(test_hashes):,}")

    print("\n--- Leakage Detection Results ---")
    print(f"  Train ∩ Val Leakage  : {len(tv_leak)} hashes")
    print(f"  Train ∩ Test Leakage : {len(tt_leak)} hashes")
    print(f"  Val ∩ Test Leakage   : {len(vt_leak)} hashes")

    total_leak = len(tv_leak) + len(tt_leak) + len(vt_leak)

    if total_leak == 0:
        print("\n✅ LEAKAGE AUDIT PASSED: ZERO SHA256 SPLIT LEAKAGE DETECTED!")
    else:
        print(f"\n❌ LEAKAGE AUDIT FAILED: {total_leak} HASH LEAKAGES DETECTED!")
        raise ValueError("Leakage audit failed!")

if __name__ == "__main__":
    main()
