"""
ZARI.ai — Plot All Training & Validation Error Curves

Loads training histories for Model A (Crop Router), Model B (3 Crop Classifiers),
Swin-Tiny Comparison Study, and Knowledge Distillation, prints the numerical tables,
and generates 4 high-resolution plots.

Outputs saved to:
  - Artifacts Directory: /home/hammad/.gemini/antigravity-ide/brain/934a9c51-6e53-48a4-abca-5a417ae198a7/
  - Repo Figures Directory: ml_pipeline/reports/figures/
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
REPO_ROOT = Path("/home/hammad/Desktop/project zari - experimental")
DATA_DIR = REPO_ROOT / "ml_pipeline" / "data"
REPORTS_V3_DIR = DATA_DIR / "reports_v3"
SWIN_DIR = REPO_ROOT / "ml_pipeline" / "models" / "swin_comparison"
DISTILLED_DIR = REPO_ROOT / "ml_pipeline" / "models" / "distilled"

ARTIFACTS_DIR = Path("/home/hammad/.gemini/antigravity-ide/brain/934a9c51-6e53-48a4-abca-5a417ae198a7")
FIG_DIR = REPO_ROOT / "ml_pipeline" / "reports" / "figures"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set Matplotlib Style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
    'figure.autolayout': True
})

def main():
    print("=" * 75)
    print("  ZARI.ai — TRAINING & VALIDATION ERROR CURVES ENGINE")
    print("=" * 75)
    
    # ── 1. Load Model A Router History ──────────────────────────────────────────
    df_a = pd.read_csv(REPORTS_V3_DIR / "model_a_training_history.csv")
    print("\n  [1/4] Model A (Crop Router - EfficientNetV2-B2) Epochs:")
    for idx, r in df_a.iterrows():
        print(f"    Epoch {int(r['epoch']):02d} | Train Loss: {r['train_loss']:.4f} | Val Loss: {r['val_loss']:.4f} | Train Acc: {r['train_accuracy']*100:.2f}% | Val Acc: {r['val_accuracy']*100:.2f}% | Val F1: {r['val_macro_f1']:.4f}")
        
    # ── Plot 1: Model A Router ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df_a['epoch'], df_a['train_loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(df_a['epoch'], df_a['val_loss'], 's--', label='Validation Loss', color='#d62728', linewidth=2)
    ax1.axvline(x=3.5, color='gray', linestyle=':', label='Stage 2 Fine-Tuning Start')
    ax1.set_title('Model A Crop Router — Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(loc='upper right')
    
    ax2.plot(df_a['epoch'], df_a['train_accuracy']*100, 'o-', label='Train Accuracy (%)', color='#2ca02c', linewidth=2)
    ax2.plot(df_a['epoch'], df_a['val_accuracy']*100, 's--', label='Val Accuracy (%)', color='#ff7f0e', linewidth=2)
    ax2.plot(df_a['epoch'], df_a['val_macro_f1']*100, '^-.', label='Val Macro F1 (%)', color='#9467bd', linewidth=2)
    ax2.axvline(x=3.5, color='gray', linestyle=':')
    ax2.set_title('Model A Crop Router — Accuracy & F1 Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend(loc='lower right')
    
    plt.suptitle('Model A EfficientNetV2-B2 Crop Router Training Metrics', y=1.02)
    fig1_path_art = ARTIFACTS_DIR / "01_model_a_crop_router_curves.png"
    fig1_path_fig = FIG_DIR / "01_model_a_crop_router_curves.png"
    fig.savefig(fig1_path_art, dpi=300)
    fig.savefig(fig1_path_fig, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved Plot 1: {fig1_path_art.name}")

    # ── 2. Load Model B Classifiers Histories ──────────────────────────────────
    df_tom = pd.read_csv(REPORTS_V3_DIR / "model_b_tomato_training_history.csv")
    df_pot = pd.read_csv(REPORTS_V3_DIR / "model_b_potato_training_history.csv")
    df_pep = pd.read_csv(REPORTS_V3_DIR / "model_b_pepper_training_history.csv")
    
    print("\n  [2/4] Model B Crop Classifiers Summary:")
    print(f"    Tomato Final Val F1 : {df_tom['val_macro_f1'].iloc[-1]:.4f} (Best: {df_tom['val_macro_f1'].max():.4f} at epoch {df_tom['val_macro_f1'].idxmax()+1})")
    print(f"    Potato Final Val F1 : {df_pot['val_macro_f1'].iloc[-1]:.4f} (Best: {df_pot['val_macro_f1'].max():.4f} at epoch {df_pot['val_macro_f1'].idxmax()+1})")
    print(f"    Pepper Final Val F1 : {df_pep['val_macro_f1'].iloc[-1]:.4f} (Best: {df_pep['val_macro_f1'].max():.4f} at epoch {df_pep['val_macro_f1'].idxmax()+1})")
    
    # ── Plot 2: Model B Production Classifiers ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(df_tom['epoch'], df_tom['val_loss'], 'o-', label='Tomato Val Loss', color='#e377c2', linewidth=2)
    ax1.plot(df_pot['epoch'], df_pot['val_loss'], 's-', label='Potato Val Loss', color='#8c564b', linewidth=2)
    ax1.plot(df_pep['epoch'], df_pep['val_loss'], '^-', label='Pepper Val Loss', color='#17becf', linewidth=2)
    ax1.set_title('Model B EfficientNet — Validation EDL Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dirichlet EDL Loss')
    ax1.legend(loc='upper right')
    
    # F1 curves
    ax2.plot(df_tom['epoch'], df_tom['val_macro_f1'], 'o-', label='Tomato Val F1', color='#e377c2', linewidth=2)
    ax2.plot(df_pot['epoch'], df_pot['val_macro_f1'], 's-', label='Potato Val F1', color='#8c564b', linewidth=2)
    ax2.plot(df_pep['epoch'], df_pep['val_macro_f1'], '^-', label='Pepper Val F1', color='#17becf', linewidth=2)
    ax2.set_title('Model B EfficientNet — Validation Macro F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1 Score')
    ax2.legend(loc='lower right')
    
    plt.suptitle('Model B Crop-Specific EDL Classifiers Training Trajectories', y=1.02)
    fig2_path_art = ARTIFACTS_DIR / "02_model_b_disease_classifiers_curves.png"
    fig2_path_fig = FIG_DIR / "02_model_b_disease_classifiers_curves.png"
    fig.savefig(fig2_path_art, dpi=300)
    fig.savefig(fig2_path_fig, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved Plot 2: {fig2_path_art.name}")

    # ── 3. Load Swin-Tiny Comparison Histories ─────────────────────────────────
    swin_tom = pd.read_csv(SWIN_DIR / "swin_tomato_training_history.csv")
    swin_pot = pd.read_csv(SWIN_DIR / "swin_potato_training_history.csv")
    swin_pep = pd.read_csv(SWIN_DIR / "swin_pepper_training_history.csv")
    
    # ── Plot 3: Swin vs EfficientNet Comparison ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    # Tomato
    axes[0].plot(df_tom['epoch'], df_tom['val_macro_f1'], 'o-', label='EfficientNetV2-B2', color='#1f77b4', linewidth=2)
    axes[0].plot(swin_tom['epoch'], swin_tom['val_macro_f1'], 's--', label='Swin-Tiny', color='#ff7f0e', linewidth=2)
    axes[0].set_title('Tomato (13 Classes)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Macro F1')
    axes[0].legend(loc='lower right')
    
    # Potato
    axes[1].plot(df_pot['epoch'], df_pot['val_macro_f1'], 'o-', label='EfficientNetV2-B2', color='#1f77b4', linewidth=2)
    axes[1].plot(swin_pot['epoch'], swin_pot['val_macro_f1'], 's--', label='Swin-Tiny', color='#ff7f0e', linewidth=2)
    axes[1].set_title('Potato (3 Classes)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='lower right')
    
    # Pepper
    axes[2].plot(df_pep['epoch'], df_pep['val_macro_f1'], 'o-', label='EfficientNetV2-B2', color='#1f77b4', linewidth=2)
    axes[2].plot(swin_pep['epoch'], swin_pep['val_macro_f1'], 's--', label='Swin-Tiny', color='#ff7f0e', linewidth=2)
    axes[2].set_title('Pepper (6 Classes)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend(loc='lower right')
    
    plt.suptitle('Swin-Tiny vs. EfficientNetV2-B2 Validation Macro F1 Comparison', y=1.02)
    fig3_path_art = ARTIFACTS_DIR / "03_swin_vs_efficientnet_f1_trajectories.png"
    fig3_path_fig = FIG_DIR / "03_swin_vs_efficientnet_f1_trajectories.png"
    fig.savefig(fig3_path_art, dpi=300)
    fig.savefig(fig3_path_fig, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved Plot 3: {fig3_path_art.name}")

    # ── 4. Load Knowledge Distillation History ──────────────────────────────────
    with open(DISTILLED_DIR / "distillation_history.json") as f:
        dist_h = json.load(f)
    df_dist = pd.DataFrame(dist_h)
    
    print("\n  [4/4] Knowledge Distillation (Swin-Tiny Teacher -> EfficientNet Student) Epochs:")
    for idx, r in df_dist.iterrows():
        print(f"    Epoch {int(r['epoch']):02d} | Train Loss: {r['train_total_loss']:.4f} (KL: {r['train_kl_loss']:.4f}, CE: {r['train_ce_loss']:.4f}) | Val Loss: {r['val_loss']:.4f} | Val F1: {r['val_macro_f1']:.4f}")
        
    # ── Plot 4: Distillation Curves ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df_dist['epoch'], df_dist['train_total_loss'], 'o-', label='Train Total Loss', color='#2ca02c', linewidth=2)
    ax1.plot(df_dist['epoch'], df_dist['train_kl_loss'], 'v:', label='Train KL Soft Loss', color='#9467bd', linewidth=1.5)
    ax1.plot(df_dist['epoch'], df_dist['train_ce_loss'], 'x:', label='Train Hard CE Loss', color='#8c564b', linewidth=1.5)
    ax1.plot(df_dist['epoch'], df_dist['val_loss'], 's--', label='Validation Loss', color='#d62728', linewidth=2)
    ax1.set_title('Knowledge Distillation — Loss Deconstruction')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Distillation Loss')
    ax1.legend(loc='upper right')
    
    ax2.plot(df_dist['epoch'], df_dist['val_macro_f1'], 'o-', label='Distilled Student Val F1', color='#1f77b4', linewidth=2)
    ax2.axhline(y=0.9787, color='#2ca02c', linestyle='--', label='Production Model B F1 (0.9787)')
    ax2.axhline(y=0.9804, color='#ff7f0e', linestyle=':', label='Swin-Tiny Teacher F1 (0.9804)')
    ax2.set_title('Knowledge Distillation — Student F1 vs Teacher & Baseline')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1 Score')
    ax2.legend(loc='lower right')
    
    plt.suptitle('Knowledge Distillation Training & Validation Trajectories', y=1.02)
    fig4_path_art = ARTIFACTS_DIR / "04_knowledge_distillation_curves.png"
    fig4_path_fig = FIG_DIR / "04_knowledge_distillation_curves.png"
    fig.savefig(fig4_path_art, dpi=300)
    fig.savefig(fig4_path_fig, dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved Plot 4: {fig4_path_art.name}")
    
    print(f"\n{'='*75}")
    print("✓ All 4 training & validation error plots generated successfully!")
    print("=" * 75)

if __name__ == "__main__":
    main()
