import os
import json
from pathlib import Path
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "mlops" else SCRIPT_DIR.parent / "ml_pipeline"
ML_PIPELINE_DIR = REPO_ROOT / "ml_pipeline"

HISTORY_JSON_PATH = ML_PIPELINE_DIR / "logs" / "phase2_training_history.json"
SCRC_JSON_PATH = ML_PIPELINE_DIR / "models" / "scrc_threshold.json"
MLRUNS_DIR = ML_PIPELINE_DIR / "mlruns"

def main():
    print("=====================================================================")
    print("  ZARI.ai — MLflow Historical Training Run Backfill")
    print("=====================================================================\n")

    if not HISTORY_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing history JSON at {HISTORY_JSON_PATH}")

    mlflow.set_tracking_uri("file:" + str(MLRUNS_DIR))
    mlflow.set_experiment("zari-phase2")

    with mlflow.start_run(run_name="phase2_cnn_baseline_historical"):
        mlflow.set_tag("source", "backfilled_from_existing_run")
        mlflow.log_params({
            "backbone": "tf_efficientnetv2_s.in21k_ft_in1k",
            "NUM_CLASSES": 67,
            "backfilled_from": str(HISTORY_JSON_PATH)
        })

        with open(HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)

        epochs = history.get("epoch", [])
        num_steps = len(epochs)

        for step_idx in range(num_steps):
            step_val = epochs[step_idx]
            metrics_step = {}
            for k, val_list in history.items():
                if k != "epoch" and isinstance(val_list, list) and step_idx < len(val_list):
                    metrics_step[k] = val_list[step_idx]
            
            mlflow.log_metrics(metrics_step, step=step_val)

        print(f"✓ Backfilled {num_steps} epochs from {HISTORY_JSON_PATH.name}")

        # Log SCRC Threshold metrics if present
        if SCRC_JSON_PATH.exists():
            with open(SCRC_JSON_PATH, "r", encoding="utf-8") as f:
                scrc_data = json.load(f)

            scrc_metrics = {}
            for k, v in scrc_data.items():
                if v is not None and isinstance(v, (int, float)):
                    scrc_metrics[f"scrc_{k}"] = float(v)

            if scrc_metrics:
                mlflow.log_metrics(scrc_metrics)
                print(f"✓ Logged {len(scrc_metrics)} SCRC threshold metrics from {SCRC_JSON_PATH.name}")

        # Log artifacts
        if HISTORY_JSON_PATH.exists():
            mlflow.log_artifact(str(HISTORY_JSON_PATH))
        if SCRC_JSON_PATH.exists():
            mlflow.log_artifact(str(SCRC_JSON_PATH))

        best_model_path = ML_PIPELINE_DIR / "models" / "phase2_best.pth"
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path))

    print("\n✅ Historical training run successfully backfilled into MLflow experiment 'zari-phase2'!")

if __name__ == "__main__":
    main()
