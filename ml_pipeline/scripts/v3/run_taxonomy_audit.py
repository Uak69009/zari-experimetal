import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "ml_pipeline" / "data"

V3_CSV_PATH = DATA_DIR / "dataset_final_training_v3.csv"

def main():
    print("=====================================================================")
    print("  ZARI.ai — Master Dataset V3 Taxonomy Harmonization Audit")
    print("=====================================================================\n")

    if not V3_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing V3 CSV at {V3_CSV_PATH}")

    df = pd.read_csv(V3_CSV_PATH, low_memory=False)

    class_names = df["class_name"].dropna().unique()
    print(f"Total Unique Classes in V3: {len(class_names):,}")

    lower_map = {}
    duplicates = []

    for cname in class_names:
        low = cname.lower().replace("-", "_").replace(" ", "_")
        if low in lower_map:
            duplicates.append((cname, lower_map[low]))
        else:
            lower_map[low] = cname

    print("\n--- Taxonomy Case & Format Variant Audit ---")
    if len(duplicates) == 0:
        print("  ✓ Zero formatting or casing duplicate class variants found.")
    else:
        print(f"  ❌ Found {len(duplicates)} duplicate taxonomy variants:")
        for c1, c2 in duplicates:
            print(f"     - '{c1}' vs '{c2}'")

    # Check Crop-Class prefix match
    mismatches = []
    for _, row in df.iterrows():
        crop = str(row["crop"])
        cname = str(row["class_name"])
        if not cname.startswith(crop + "_"):
            mismatches.append((crop, cname))

    print("\n--- Crop vs Class Prefix Alignment Audit ---")
    if len(mismatches) == 0:
        print("  ✓ 100% of class names match their assigned crop prefix.")
    else:
        print(f"  ❌ Found {len(mismatches)} crop-class prefix mismatches!")

    if len(duplicates) == 0 and len(mismatches) == 0:
        print("\n✅ TAXONOMY AUDIT PASSED: ALL TAXONOMY RULES SATISFIED!")
    else:
        raise ValueError("Taxonomy audit failed!")

if __name__ == "__main__":
    main()
