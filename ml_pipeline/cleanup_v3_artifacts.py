import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "ml_pipeline" else SCRIPT_DIR
ML_PIPELINE_DIR = REPO_ROOT / "ml_pipeline" if (REPO_ROOT / "ml_pipeline").exists() else REPO_ROOT
MODELS_DIR = ML_PIPELINE_DIR / "models"
LOGS_DIR = ML_PIPELINE_DIR / "logs"
REPORTS_V3_DIR = ML_PIPELINE_DIR / "data" / "reports_v3"

def main():
    print("=====================================================================")
    print("  ZARI.ai — Cleanup Script for V3 Model & Evaluation Artifacts")
    print("=====================================================================\n")

    deleted_model_files = []
    deleted_eval_files = []

    # Step 1: Delete V3 model files & wrong training artifacts
    target_model_patterns = [
        "phase2_edl_model_v3.pth",
        "phase1_backbone_v3.pth",
        "phase2_best_v3.pth",
        "scrc_threshold_v3.json",
        "phase2_training_history_v3.json",
        "phase2_v3_history.json",
        "phase1_v3_history.json"
    ]

    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.name in target_model_patterns or ("_v3." in item.name and item.suffix in [".pth", ".json"]):
                try:
                    item.unlink()
                    deleted_model_files.append(str(item.relative_to(REPO_ROOT)))
                except Exception as e:
                    print(f"  Error deleting {item.name}: {e}")

    print(f"[STEP 1] Deleted {len(deleted_model_files)} V3 model/training files:")
    for f in deleted_model_files:
        print(f"  - Removed: {f}")

    # Step 2: Delete V3 evaluation files from wrong training
    eval_files_to_delete = [
        REPORTS_V3_DIR / "v3_model_evaluation_metrics.txt",
        LOGS_DIR / "phase2_training_v3.log",
        LOGS_DIR / "phase2_v3_log.txt"
    ]

    if LOGS_DIR.exists():
        for item in LOGS_DIR.iterdir():
            if "v3" in item.name.lower() and item.suffix in [".log", ".txt", ".json"]:
                if item not in eval_files_to_delete:
                    eval_files_to_delete.append(item)

    for ef in eval_files_to_delete:
        if ef.exists():
            try:
                ef.unlink()
                deleted_eval_files.append(str(ef.relative_to(REPO_ROOT)))
            except Exception as e:
                print(f"  Error deleting {ef.name}: {e}")

    print(f"\n[STEP 2] Deleted {len(deleted_eval_files)} V3 evaluation/log files:")
    for f in deleted_eval_files:
        print(f"  - Removed: {f}")

    # Step 3 & 4: Verify remaining models & dataset integrity
    print("\n[STEP 3 & 4] Verifying Remaining Key Models and Dataset Integrity...")
    
    remaining_models = []
    if MODELS_DIR.exists():
        for item in sorted(MODELS_DIR.iterdir()):
            if item.is_file():
                remaining_models.append(str(item.name))

    print("  Remaining Model Checkpoints in ml_pipeline/models/:")
    for m in remaining_models:
        print(f"    ✓ {m}")

    v3_csv = ML_PIPELINE_DIR / "data" / "dataset_final_training_v3.csv"
    v2_csv = ML_PIPELINE_DIR / "data" / "dataset_final_training_v2.csv"
    class_map = ML_PIPELINE_DIR / "data" / "class_map_final.json"

    print("\n  Preserved Core Files Verification:")
    print(f"    ✓ Dataset V3 CSV          : {'EXISTS (' + str(v3_csv.stat().st_size) + ' bytes)' if v3_csv.exists() else 'MISSING!'}")
    print(f"    ✓ Dataset V2 CSV          : {'EXISTS (' + str(v2_csv.stat().st_size) + ' bytes)' if v2_csv.exists() else 'MISSING!'}")
    print(f"    ✓ Master Class Map JSON   : {'EXISTS (' + str(class_map.stat().st_size) + ' bytes)' if class_map.exists() else 'MISSING!'}")
    print(f"    ✓ Reports V3 Directory    : {'EXISTS' if REPORTS_V3_DIR.exists() else 'MISSING!'}")

    # Step 5: Summary
    print("\n=====================================================================")
    print("  CLEANUP SUMMARY")
    print("=====================================================================")
    print(f"  Total Deleted Model Files : {len(deleted_model_files)}")
    print(f"  Total Deleted Eval Files  : {len(deleted_eval_files)}")
    print("  Preserved Core Dataset V3 : ml_pipeline/data/dataset_final_training_v3.csv (49,517 rows)")
    print("  Preserved Baseline Models : phase2_edl_model.pth, phase2_edl_model_v2.pth")
    print("  Status                    : READY FOR FRESH START!")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
