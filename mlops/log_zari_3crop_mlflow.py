"""
ZARI.ai — MLOps MLflow Logger & Experiment Tracker for 3-Crop System
Logs Model B EfficientNet metrics, Swin-Tiny comparison, ChromaDB RAG specs,
Phase 7 system validation benchmarks, and production model decisions to MLflow.
"""

import os
import json
from pathlib import Path
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ML_PIPELINE_DIR = REPO_ROOT / "ml_pipeline"
DATA_DIR = ML_PIPELINE_DIR / "data"
MLRUNS_DIR = ML_PIPELINE_DIR / "mlruns"

EFFNET_METRICS_PATH = DATA_DIR / "reports_v3" / "model_b_test_metrics.json"
SWIN_METRICS_PATH = ML_PIPELINE_DIR / "models" / "swin_comparison" / "swin_test_metrics.json"
SYSTEM_VALIDATION_PATH = DATA_DIR / "phase7_5_fixpack_results.json"
FINAL_REPORT_PATH = ML_PIPELINE_DIR / "final" / "ZARI_3CROP_FINAL_REPORT.md"

def log_to_mlflow():
    print("=" * 75)
    print("  ZARI.ai — MLOps MLflow Tracking & Integration Engine")
    print("=" * 75)
    
    mlflow.set_tracking_uri("file:" + str(MLRUNS_DIR))
    mlflow.set_experiment("zari_3crop_production_system")
    
    # ── RUN 1: Locked Production Vision Models (Model B EfficientNetV2-B2) ──
    with mlflow.start_run(run_name="production_efficientnet_model_b"):
        mlflow.set_tag("stage", "production_vision")
        mlflow.set_tag("architecture", "EfficientNetV2-B2")
        mlflow.set_tag("status", "LOCKED_PRODUCTION")
        
        mlflow.log_params({
            "model_a_router": "EfficientNetV2-B2",
            "model_b_classifiers": "EfficientNetV2-B2 (Tomato, Potato, Pepper)",
            "loss_function": "EDL_Dirichlet_LogLikelihood",
            "kl_penalty": 0.1,
            "optimizer": "AdamW",
            "head_lr": 1e-3,
            "backbone_lr": 1e-4,
            "global_scrc_th_crop": 0.85,
            "global_scrc_th_disease": 0.70,
            "global_scrc_th_unc": 0.45
        })
        
        if EFFNET_METRICS_PATH.exists():
            with open(EFFNET_METRICS_PATH) as f:
                effnet_m = json.load(f)
                
            for crop in ["Tomato", "Potato", "Pepper"]:
                test_m = effnet_m[crop]["test"]
                mlflow.log_metrics({
                    f"{crop.lower()}_test_macro_f1": test_m["macro_f1"],
                    f"{crop.lower()}_test_accuracy": test_m["acc"],
                    f"{crop.lower()}_balanced_acc": test_m["bal_acc"],
                    f"{crop.lower()}_ece": test_m["ece"],
                    f"{crop.lower()}_brier": test_m["brier"]
                })
            mlflow.log_artifact(str(EFFNET_METRICS_PATH))
            print("✓ Logged Production EfficientNetV2-B2 metrics & artifacts to MLflow.")
            
    # ── RUN 2: Swin-Tiny Comparison Study ──
    with mlflow.start_run(run_name="swin_tiny_comparison_study"):
        mlflow.set_tag("stage", "architecture_comparison")
        mlflow.set_tag("architecture", "Swin-Tiny")
        mlflow.set_tag("status", "REJECTED_FOR_PRODUCTION (Grad-CAM Incompatible)")
        
        mlflow.log_params({
            "architecture": "Swin-Tiny",
            "reason_not_adopted": "Grad-CAM out-of-the-box FAIL due to (B,H,W,C) feature tensor format"
        })
        
        if SWIN_METRICS_PATH.exists():
            with open(SWIN_METRICS_PATH) as f:
                swin_m = json.load(f)
            mlflow.log_artifact(str(SWIN_METRICS_PATH))
            print("✓ Logged Swin-Tiny comparative evaluation artifacts to MLflow.")
            
    # ── RUN 3: Full RAG & End-to-End System Validation ──
    with mlflow.start_run(run_name="system_validation_and_rag_pipeline"):
        mlflow.set_tag("stage", "full_system_validation")
        mlflow.set_tag("rag_embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
        mlflow.set_tag("vector_store", "ChromaDB Persistent Store (208 chunks)")
        mlflow.set_tag("verdict", "PRODUCTION_READY_WITH_LIMITATIONS")
        
        if SYSTEM_VALIDATION_PATH.exists():
            with open(SYSTEM_VALIDATION_PATH) as f:
                val_data = json.load(f)
                
            latency_stats = val_data.get("fix1b_real_latency", val_data.get("fix1_real_latency"))
            mlflow.log_metrics({
                "real_cuda_mean_latency_ms": latency_stats["total_end_to_end"]["mean_ms"],
                "real_cuda_median_latency_ms": latency_stats["total_end_to_end"]["median_ms"],
                "real_cuda_p90_latency_ms": latency_stats["total_end_to_end"]["p90_ms"],
                "real_vision_inference_ms": latency_stats["vision_inference"]["mean_ms"],
                "real_sam2_segmentation_ms": latency_stats["sam2_segmentation"]["mean_ms"],
                "real_chromadb_retrieval_ms": latency_stats["chroma_retrieval"]["mean_ms"],
                "blind_retrieval_accuracy_pct": val_data["fix3_blind_retrieval"]["blind_accuracy_pct"]
            })
            mlflow.log_artifact(str(SYSTEM_VALIDATION_PATH))
            print("✓ Logged System Integration & RAG Validation benchmarks to MLflow.")
            
        if FINAL_REPORT_PATH.exists():
            mlflow.log_artifact(str(FINAL_REPORT_PATH))
            print("✓ Logged Master Final Report artifact to MLflow.")
            
    print("\n✅ All ZARI 3-Crop System runs, metrics, and artifacts successfully logged to MLflow experiment 'zari_3crop_production_system'!")

if __name__ == "__main__":
    log_to_mlflow()
