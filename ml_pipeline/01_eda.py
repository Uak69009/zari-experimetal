"""
01_eda.py — Advanced Exploratory Data Analysis (EDA)

Implements a 5-Phase EDA Pipeline:
Phase 1: Data Integrity (Deep decoding check)
Phase 2: Class Distribution (Plotting class balance)
Phase 3: Dimensionality & Properties (Sampling dimensions & color spaces)
Phase 4: Visual Sampling (Saving a grid of random images)
Phase 5: Strategy Handoff (Calculating inverse class weights and saving to JSON)
"""

import os
import json
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASET_PATH = r"D:\New folder\zari\zari-ai\ml_pipeline\data\raw\plantvillage\raw\color"
EDA_ARTIFACTS_DIR = r"D:\New folder\zari\zari-ai\ml_pipeline\eda_artifacts"
SUMMARY_JSON_PATH = r"D:\New folder\zari\zari-ai\ml_pipeline\eda_summary.json"

# Create artifacts directory if it doesn't exist
os.makedirs(EDA_ARTIFACTS_DIR, exist_ok=True)

def get_all_images(data_dir):
    """Utility to get all image paths and their corresponding labels."""
    image_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_paths.append(os.path.join(cls_dir, img_name))
                labels.append(cls)
                
    return image_paths, labels, classes

def phase1_data_integrity(image_paths, sample_size=5000):
    """
    Phase 1: Data Integrity
    Uses .verify() and .transpose() to check for truncated or corrupt image bytes.
    (Sampling 5000 random images here to save time, but can be run on all)
    """
    print("\n--- Phase 1: Data Integrity ---")
    sample_paths = random.sample(image_paths, min(len(image_paths), sample_size))
    corrupt_files = []
    
    for img_path in tqdm(sample_paths, desc="Verifying Image Integrity"):
        try:
            # Basic header check
            with Image.open(img_path) as img:
                img.verify()
            # Deep decoding check (forces reading all bytes)
            with Image.open(img_path) as img:
                img.transpose(Image.FLIP_LEFT_RIGHT)
        except Exception as e:
            corrupt_files.append((img_path, str(e)))
            
    print(f"Verified {len(sample_paths)} images. Found {len(corrupt_files)} corrupt files.")
    if corrupt_files:
        print("Example corrupt file:", corrupt_files[0])
    return corrupt_files

def phase2_class_distribution(labels):
    """
    Phase 2: Class Distribution
    Computes frequencies and saves a horizontal bar plot.
    """
    print("\n--- Phase 2: Class Distribution ---")
    class_counts = Counter(labels)
    df = pd.DataFrame(class_counts.items(), columns=["Class", "Count"])
    df = df.sort_values(by="Count", ascending=True) # Ascending for horizontal bar plot
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(df["Class"], df["Count"], color='skyblue')
    plt.xlabel('Number of Images')
    plt.title('Class Distribution in PlantVillage Dataset')
    plt.tight_layout()
    
    plot_path = os.path.join(EDA_ARTIFACTS_DIR, "class_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Class distribution plot saved to: {plot_path}")
    
    return class_counts

def phase3_dimensionality(image_paths, sample_size=1000):
    """
    Phase 3: Dimensionality & Properties
    Samples random images to measure WxH, aspect ratios, and color spaces.
    """
    print("\n--- Phase 3: Dimensionality & Properties ---")
    sample_paths = random.sample(image_paths, min(len(image_paths), sample_size))
    
    widths, heights, modes = [], [], []
    
    for img_path in tqdm(sample_paths, desc="Analyzing Dimensions"):
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                modes.append(img.mode)
        except:
            continue
            
    avg_w, avg_h = np.mean(widths), np.mean(heights)
    mode_counts = Counter(modes)
    
    print(f"Sampled {len(widths)} images for dimensionality.")
    print(f"Average Dimensions : {avg_w:.1f} W x {avg_h:.1f} H")
    print(f"Min Dimensions     : {min(widths)} W x {min(heights)} H")
    print(f"Max Dimensions     : {max(widths)} W x {max(heights)} H")
    print(f"Color Spaces (Modes): {dict(mode_counts)}")
    
    if min(widths) == max(widths) and min(heights) == max(heights):
        print("Conclusion: All images have uniform dimensions. Direct resizing is safe.")
    else:
        print("Conclusion: Images have varying dimensions. Consider center-cropping or padding.")

def phase4_visual_sampling(image_paths, labels, classes):
    """
    Phase 4: Visual Sampling
    Generates a 4x4 Matplotlib grid containing random images across distinct classes.
    """
    print("\n--- Phase 4: Visual Sampling ---")
    plt.figure(figsize=(12, 12))
    
    # Pick 16 random unique classes (or as many as available)
    sample_classes = random.sample(classes, min(16, len(classes)))
    
    for i, cls in enumerate(sample_classes):
        # Find one random image belonging to this class
        cls_images = [img for img, lbl in zip(image_paths, labels) if lbl == cls]
        img_path = random.choice(cls_images)
        
        try:
            img = Image.open(img_path).convert("RGB")
            plt.subplot(4, 4, i + 1)
            plt.imshow(img)
            # Make label readable (e.g. "Apple___Apple_scab" -> "Apple scab")
            readable_label = cls.split("___")[-1].replace("_", " ")
            plt.title(readable_label[:20], fontsize=10)
            plt.axis('off')
        except:
            pass
            
    plt.tight_layout()
    plot_path = os.path.join(EDA_ARTIFACTS_DIR, "visual_sample_grid.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Visual sample grid saved to: {plot_path}")

def phase5_strategy_handoff(class_counts):
    """
    Phase 5: Strategy Handoff
    Computes inverse class weights for balanced training and saves to JSON.
    Formula: Weight_c = N_total / (N_classes * N_c)
    """
    print("\n--- Phase 5: Strategy Handoff ---")
    N_total = sum(class_counts.values())
    N_classes = len(class_counts)
    
    class_weights = {}
    for cls, N_c in class_counts.items():
        weight = N_total / (N_classes * N_c)
        class_weights[cls] = round(weight, 4)
        
    summary_data = {
        "dataset_stats": {
            "total_images": N_total,
            "total_classes": N_classes
        },
        "class_weights": class_weights
    }
    
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"Computed inverse class weights.")
    print(f"Strategy summary saved to: {SUMMARY_JSON_PATH}")

def run_pipeline():
    print("==================================================")
    print("  ZARI.ai — 5-PHASE EXPLORATORY DATA ANALYSIS")
    print("==================================================")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
        
    image_paths, labels, classes = get_all_images(DATASET_PATH)
    print(f"Discovered {len(image_paths)} total images across {len(classes)} classes.")
    
    # Execute the 5 phases
    phase1_data_integrity(image_paths)
    class_counts = phase2_class_distribution(labels)
    phase3_dimensionality(image_paths)
    phase4_visual_sampling(image_paths, labels, classes)
    phase5_strategy_handoff(class_counts)
    
    print("\n==================================================")
    print("EDA Pipeline Complete. You are ready for Phase 2 (Training).")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()
