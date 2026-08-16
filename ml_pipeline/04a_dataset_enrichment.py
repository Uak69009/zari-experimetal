"""
ZARI.ai Dataset Enrichment Pipeline (Stage 1)
---------------------------------------------
This script reads the master dataset CSV, extracts complex taxonomies,
infers domains, and runs a multiprocessing pool to calculate computer vision
metrics (blur, brightness, contrast, entropy) for all 251,000+ images.

WARNING: Running this on 250K images will take a significant amount of time.
"""

import os
import cv2
import pandas as pd
import numpy as np
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import math

# ==========================================
# TAXONOMY MAPPINGS
# ==========================================
CROP_FAMILY_MAP = {
    "Apple": "Rosaceae",
    "Corn": "Poaceae",
    "Grape": "Vitaceae",
    "Peach": "Rosaceae",
    "Pepper": "Solanaceae",
    "Potato": "Solanaceae",
    "Strawberry": "Rosaceae",
    "Tomato": "Solanaceae",
    "Citrus": "Rutaceae",
    "Cherry": "Rosaceae",
    "Soybean": "Fabaceae",
    "Squash": "Cucurbitaceae",
    "Raspberry": "Rosaceae",
    "Blueberry": "Ericaceae",
    "Wheat": "Poaceae",
    "Rice": "Poaceae"
}

def infer_taxonomy(unified_label):
    # e.g. "Apple_Scab" or "Tomato_Late_Blight"
    parts = unified_label.split("_")
    crop = parts[0]
    disease = "_".join(parts[1:])
    
    crop_family = CROP_FAMILY_MAP.get(crop, "Unknown")
    
    pathogen_type = "Unknown"
    disease_lower = disease.lower()
    
    if "healthy" in disease_lower:
        pathogen_type = "Healthy"
    elif any(v in disease_lower for v in ["virus", "mosaic", "curl", "yellow"]):
        pathogen_type = "Viral"
    elif any(b in disease_lower for b in ["bacterial", "pseudomonas", "xanthomonas", "canker"]):
        pathogen_type = "Bacterial"
    elif any(f in disease_lower for f in ["rust", "blight", "scab", "mold", "mildew", "rot", "smut", "fungus", "spot"]):
        pathogen_type = "Fungal"
    elif any(p in disease_lower for p in ["mite", "insect", "miner", "bug", "spider"]):
        pathogen_type = "Pest"
    elif any(n in disease_lower for n in ["deficiency", "nutrient"]):
        pathogen_type = "Nutrient"
        
    return crop, disease, crop_family, pathogen_type

# ==========================================
# COMPUTER VISION METRICS
# ==========================================
def compute_image_metrics(image_path):
    metrics = {
        "image_width": None,
        "image_height": None,
        "aspect_ratio": None,
        "blur_score": None,
        "brightness_score": None,
        "contrast_score": None,
        "sharpness_score": None,
        "noise_score": None,
        "entropy_score": None,
        "background_complexity": None,
        "edge_density": None,
        "image_quality_score": None,
        "difficulty_score": "Unknown",
        "is_corrupt": True
    }
    
    if not os.path.exists(image_path):
        return metrics
        
    if os.path.getsize(image_path) == 0:
        return metrics
        
    try:
        # Load image in grayscale for fast metric computation
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return metrics
            
        h, w = img_gray.shape
        metrics["image_height"] = h
        metrics["image_width"] = w
        metrics["aspect_ratio"] = round(w / h, 2) if h > 0 else 0
        
        # Blur (Laplacian variance)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        blur_score = laplacian.var()
        metrics["blur_score"] = round(blur_score, 2)
        
        # Brightness & Contrast
        mean, stddev = cv2.meanStdDev(img_gray)
        brightness = mean[0][0]
        contrast = stddev[0][0]
        metrics["brightness_score"] = round(brightness, 2)
        metrics["contrast_score"] = round(contrast, 2)
        
        # Sharpness (Max gradient magnitude)
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        metrics["sharpness_score"] = round(np.max(grad_mag), 2)
        
        # Edges & Background Complexity
        edges = cv2.Canny(img_gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (w * h)
        metrics["edge_density"] = round(edge_density, 4)
        metrics["background_complexity"] = round(edge_density * 100, 2)
        
        # Entropy (Shannon)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        non_zero_hist = hist[hist > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
        metrics["entropy_score"] = round(entropy, 2)
        
        # Noise Proxy (Standard deviation of Laplacian)
        metrics["noise_score"] = round(np.std(laplacian), 2)
        
        # Synthetic Quality Score (Heuristic: 0-100)
        brightness_penalty = abs(brightness - 127) / 127.0
        quality = (contrast * 0.4) + (min(blur_score, 1000) / 1000 * 50) - (brightness_penalty * 20)
        quality = max(0, min(100, quality))
        metrics["image_quality_score"] = round(quality, 2)
        
        # Difficulty Score
        if blur_score < 50 or edge_density > 0.15 or brightness < 40 or brightness > 220:
            metrics["difficulty_score"] = "Hard"
        elif blur_score < 150 or edge_density > 0.10:
            metrics["difficulty_score"] = "Medium"
        else:
            metrics["difficulty_score"] = "Easy"
            
        metrics["is_corrupt"] = False
        
    except Exception as e:
        pass
        
    return metrics

def process_row(row_dict):
    """Processes a single row for the DataFrame in a worker thread."""
    img_path = row_dict["image_path"]
    
    # 1. Taxonomies
    crop, disease, crop_family, pathogen = infer_taxonomy(row_dict["unified_label"])
    
    # 2. Domain & Annotation Type
    source = str(row_dict.get("dataset_source", "Unknown")).lower()
    domain = "Mixed"
    if "plantvillage" in source:
        domain = "Lab"
    elif "plantdoc" in source or "nwrd" in source:
        domain = "Field"
        
    annotation = "classification"
    if "segmented" in img_path.lower() or "mask" in img_path.lower():
        annotation = "classification+segmentation"
        
    # 3. OpenCV Metrics
    metrics = compute_image_metrics(img_path)
    
    # Merge everything
    enriched_row = {
        "image_path": img_path,
        "crop": crop,
        "disease": disease,
        "class_name": row_dict["unified_label"],
        "class_id": None, # Will be assigned sequentially later
        "split": row_dict.get("split", "train"),
        "source_dataset": source,
        "domain": domain,
        "annotation_type": annotation,
        
        "crop_family": crop_family,
        "disease_family": disease,
        "pathogen_type": pathogen,
        
        # Severity placeholders
        "lesion_pixels": None,
        "leaf_pixels": None,
        "lesion_percentage": None,
        "severity_score": None,
        "severity_class": "Unknown",
        
        # Future placeholders
        "weather": None, "temperature": None, "humidity": None, "gps": None, 
        "country": None, "camera_type": None, "timestamp": None, 
        "farm_id": None, "farmer_id": None
    }
    
    enriched_row.update(metrics)
    return enriched_row


def main():
    print("=====================================================")
    print("ZARI.ai Research Dataset Enrichment Pipeline")
    print("=====================================================")
    
    # Resolving absolute paths based on current working directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    master_csv_path = os.path.join(base_dir, "data", "dataset_master.csv")
    output_csv_path = os.path.join(base_dir, "data", "dataset_master_enriched.csv")
    
    if not os.path.exists(master_csv_path):
        print(f"Error: {master_csv_path} not found.")
        return
        
    print(f"Loading master CSV: {master_csv_path}")
    df = pd.read_csv(master_csv_path)
    
    # Check if dataset is huge. If we want a fast run, uncomment slice
    # print("Slicing for fast testing... Processing 1000 rows.")
    # df = df.head(1000)
    
    # Create unique class IDs
    unique_classes = sorted(df["unified_label"].unique())
    class_id_map = {name: idx for idx, name in enumerate(unique_classes)}
    
    rows = df.to_dict('records')
    
    # MULTIPROCESSING -> THREADING (Safer for OpenCV on Windows, no BrokenProcessPool)
    print(f"Processing {len(rows)} images using OpenCV Threading...")
    max_workers = os.cpu_count() or 4
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers * 2) as executor:
        results = list(tqdm(executor.map(process_row, rows), total=len(rows), desc="Extracting Metrics"))
        
    # Filter corrupted
    corrupt_count = sum(1 for r in results if r.get("is_corrupt", True))
    valid_results = [r for r in results if not r.get("is_corrupt", True)]
    
    print(f"\nCompleted! Found {corrupt_count} missing/corrupted/zero-byte files.")
    
    # Assign Class IDs
    for r in valid_results:
        r["class_id"] = class_id_map.get(r["class_name"], -1)
        if "is_corrupt" in r:
            del r["is_corrupt"]
            
    # Save enriched CSV
    enriched_df = pd.DataFrame(valid_results)
    
    # Validation checks
    dup_paths = enriched_df["image_path"].duplicated().sum()
    print(f"Validation: Found {dup_paths} duplicate image paths.")
    if dup_paths > 0:
        enriched_df.drop_duplicates(subset=["image_path"], inplace=True)
        print("Dropped duplicate paths.")
        
    null_labels = enriched_df["class_name"].isnull().sum()
    print(f"Validation: Found {null_labels} missing labels.")
    
    enriched_df.to_csv(output_csv_path, index=False)
    print(f"\nEnriched dataset successfully saved to: {output_csv_path}")
    print(f"Total Rows Saved: {len(enriched_df)}")

if __name__ == "__main__":
    main()
