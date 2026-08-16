"""
02_data_harmonization.py — Master Data Harmonization & Indexing

OBJECTIVES:
1. Deep-scan all image folders in ml_pipeline/data/raw/
2. Smart Label Normalization & Harmonization (regex-based).
3. Human Review Gate for mappings.
4. Stratified Train/Val/Test split (80/10/10) using scikit-learn.
5. Export dataset_master.csv.
"""

import os
import re
import pandas as pd
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split

DATA_RAW_DIR = r"D:\New folder\zari\zari-ai\ml_pipeline\data\raw"
CSV_OUTPUT_PATH = r"D:\New folder\zari\zari-ai\ml_pipeline\data\dataset_master.csv"

# Comprehensive crop mapping for strict Crop + Disease isolation
CROP_MAP = {
    'apple': 'Apple',
    'blueberry': 'Blueberry',
    'cherry': 'Cherry',
    'corn': 'Corn',
    'maize': 'Corn',
    'grape': 'Grape',
    'orange': 'Orange',
    'citrus': 'Orange',
    'peach': 'Peach',
    'pepper': 'Pepper',
    'bell_pepper': 'Pepper',
    'potato': 'Potato',
    'raspberry': 'Raspberry',
    'soybean': 'Soybean',
    'soyabean': 'Soybean',
    'squash': 'Squash',
    'strawberry': 'Strawberry',
    'tomato': 'Tomato',
    'bean': 'Bean',
    'walnut': 'Walnut',
    'pear': 'Pear',
    'apricot': 'Apricot',
    'fig': 'Fig'
}

NOISE_WORDS = {
    'leaf', 'disease', 'image', 'photo', 'normal', 'crop', 'plant', 
    'including', 'sour', 'maize', 'bell', 'spot'
}

def normalize_label(raw_label):
    s = raw_label.lower()
    
    # Identify the crop
    crop = "Unknown"
    for k, v in CROP_MAP.items():
        if k in s:
            crop = v
            break
            
    # Clean up non-alphanumeric chars into spaces FIRST so word boundaries work
    disease_str = re.sub(r'[^a-z0-9]', ' ', s)
    
    # If crop is found, strip out all crop synonyms from the string to isolate disease
    for k in CROP_MAP.keys():
        disease_str = re.sub(r'\b' + k + r'\b', ' ', disease_str)
        
    # Strip noise words
    for w in NOISE_WORDS:
        # Note: 'spot' is tricky (Bacterial_spot), but the user said remove 'leaf', 'disease', etc.
        # Let's selectively remove noise words safely
        if w != 'spot': # Keep spot for 'bacterial spot'
            disease_str = re.sub(r'\b' + w + r'\b', ' ', disease_str)
            
    disease_words = [w for w in disease_str.split() if w.strip()]
    
    if 'healthy' in disease_words:
        return f"{crop}_Healthy"
        
    if not disease_words:
        return f"{crop}_Unknown"
        
    # Capitalize each word in the disease name and join with underscores
    disease = "_".join([w.capitalize() for w in disease_words])
    
    return f"{crop}_{disease}"


def scan_datasets():
    print(f"Scanning raw datasets in {DATA_RAW_DIR}...")
    dataset_records = []
    
    for dataset_name in sorted(os.listdir(DATA_RAW_DIR)):
        dataset_path = os.path.join(DATA_RAW_DIR, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
            
        for root, dirs, files in os.walk(dataset_path):
            # The parent folder of the files is typically the raw class label
            raw_label = os.path.basename(root)
            
            # Optimization: If there are no images in this folder, skip.
            # But we process file by file to build the full dataset index.
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(root, file)
                    unified_label = normalize_label(raw_label)
                    
                    dataset_records.append({
                        "image_path": image_path,
                        "dataset_source": dataset_name,
                        "raw_label": raw_label,
                        "unified_label": unified_label
                    })
                    
    return pd.DataFrame(dataset_records)


def perform_stratified_split(df):
    print("\nPerforming Stratified 80/10/10 Split...")
    
    # 1. Identify classes with < 3 samples (scikit-learn stratify requires >= 2 per split ideally, but we need 3 for train/val/test)
    class_counts = df['unified_label'].value_counts()
    rare_classes = class_counts[class_counts < 3].index.tolist()
    
    if rare_classes:
        print(f"WARNING: Found {len(rare_classes)} classes with < 3 samples. Assigning entirely to TRAIN.")
        for rc in rare_classes:
            print(f"  - {rc} ({class_counts[rc]} samples)")
            
    # Separate rare and valid classes
    df_rare = df[df['unified_label'].isin(rare_classes)].copy()
    df_valid = df[~df['unified_label'].isin(rare_classes)].copy()
    
    df_rare['split'] = 'train'
    
    if df_valid.empty:
        return df_rare
        
    # Split valid classes: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df_valid, 
        test_size=0.20, 
        stratify=df_valid['unified_label'], 
        random_state=42
    )
    
    # Split temp: 50% val, 50% test (which equals 10% / 10% of total)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['unified_label'], 
        random_state=42
    )
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine back together
    final_df = pd.concat([train_df, val_df, test_df, df_rare], ignore_index=True)
    return final_df


def run_harmonization(review_only=False):
    print("=" * 60)
    print("  ZARI.ai — DATA HARMONIZATION & INDEXING")
    print("=" * 60)
    
    df = scan_datasets()
    
    if df.empty:
        print("No images found to process.")
        return
        
    print(f"\nDiscovered {len(df)} total images.")
    
    # Generate Mapping Table
    mapping = df[['raw_label', 'unified_label']].drop_duplicates().sort_values('unified_label')
    
    print("\n" + "=" * 60)
    print("  HUMAN REVIEW GATE: LABEL MAPPING TABLE")
    print("=" * 60)
    print(f"{'RAW LABEL'.ljust(45)} | {'UNIFIED LABEL'.ljust(35)}")
    print("-" * 85)
    
    for _, row in mapping.iterrows():
        print(f"{row['raw_label'][:43].ljust(45)} | {row['unified_label']}")
        
    print("-" * 85)
    print(f"Total Raw Classes: {df['raw_label'].nunique()}")
    print(f"Total Unified Classes: {df['unified_label'].nunique()}")
    
    if review_only:
        print("\n[REVIEW MODE] Exiting. Run without --review to finalize and split.")
        return

    # User Approval
    print("\nPlease review the mapping table above.")
    response = input("Approve mapping and proceed with data splitting? (Y/N): ")
    if response.strip().upper() != 'Y':
        print("Operation cancelled by user.")
        return
        
    # Proceed to splitting
    df_split = perform_stratified_split(df)
    
    # Output Logging
    print("\n" + "=" * 60)
    print("  SPLIT DISTRIBUTION SUMMARY")
    print("=" * 60)
    split_counts = df_split['split'].value_counts()
    print(f"Train : {split_counts.get('train', 0)} images")
    print(f"Val   : {split_counts.get('val', 0)} images")
    print(f"Test  : {split_counts.get('test', 0)} images")
    
    # Export
    # Reorder columns as requested (omitting width/height for speed)
    df_split = df_split[['image_path', 'dataset_source', 'raw_label', 'unified_label', 'split']]
    df_split.to_csv(CSV_OUTPUT_PATH, index=False)
    
    print(f"\nSUCCESS! Master dataset CSV exported to:\n  {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true", help="Print mapping table and exit without splitting")
    args = parser.parse_args()
    
    run_harmonization(review_only=args.review)
