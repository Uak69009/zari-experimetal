import os, json
import pandas as pd
import numpy as np

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXISTING_CSV = 'ml_pipeline/data/dataset_final_training.csv'
NEW_QC_CSV = 'ml_pipeline/staging_qc_passed.csv'
CLASS_MAP_FILE = 'ml_pipeline/data/class_map_final.json'

def main():
    print('=====================================================================')
    print('PHASE 7 -- SCHEMA ALIGNMENT & MERGE EXECUTION')
    print('=====================================================================\n')

    # 1. Load Existing Dataset
    if not os.path.exists(EXISTING_CSV):
        raise FileNotFoundError(f'Existing dataset file missing: {EXISTING_CSV}')
    df_exist = pd.read_csv(EXISTING_CSV, low_memory=False)
    print(f'✓ Loaded Existing Dataset ({EXISTING_CSV}): {len(df_exist)} rows, {len(df_exist.columns)} columns')

    # 2. Load QC-Passed New Data
    if not os.path.exists(NEW_QC_CSV):
        raise FileNotFoundError(f'QC passed file missing: {NEW_QC_CSV}')
    df_new = pd.read_csv(NEW_QC_CSV)
    print(f'✓ Loaded QC-Passed New Data ({NEW_QC_CSV}): {len(df_new)} rows')

    # 3. Load Class Map
    with open(CLASS_MAP_FILE) as f:
        cmap = json.load(f)
    head_classes = cmap['head_classes']
    pretrain_classes = cmap['pretrain_classes']

    # 4. Align New Data to Existing Schema
    print('\n--- Aligning New Data Schema to Existing Dataset ---')
    aligned_rows = []
    
    for idx, row in df_new.iterrows():
        w_cls = row['wheat_class']
        cid = head_classes[w_cls]
        pid = pretrain_classes.get(w_cls, -1)
        disease_name = w_cls.replace('Wheat_', '')
        
        w = row['width']
        h = row['height']
        ar = round(float(w) / float(h), 4) if h > 0 else np.nan
        
        aligned_row = {
            'image_path': row['filepath'],
            'crop': 'Wheat',
            'disease': disease_name,
            'class_name': w_cls,
            'class_id': cid,
            'split': 'train',  # Placeholder for Phase 8 re-split
            'source_dataset': row['source_dataset'],
            'domain': 'Field',
            'annotation_type': 'classification',
            'crop_family': 'Poaceae',
            'disease_family': disease_name,
            'pathogen_type': 'Fungus' if disease_name in ['Blast', 'Brown_Rust', 'Leaf_Blight'] else ('None' if disease_name == 'Healthy' else 'Unknown'),
            'lesion_pixels': np.nan,
            'leaf_pixels': np.nan,
            'lesion_percentage': np.nan,
            'severity_score': np.nan,
            'severity_class': 'Unknown',
            'weather': np.nan,
            'temperature': np.nan,
            'humidity': np.nan,
            'gps': np.nan,
            'country': 'Bangladesh' if row['source_dataset'] == 'Bangladesh' else ('Ethiopia' if row['source_dataset'] == 'CGIAR' else np.nan),
            'camera_type': np.nan,
            'timestamp': np.nan,
            'farm_id': np.nan,
            'farmer_id': np.nan,
            'image_width': w,
            'image_height': h,
            'aspect_ratio': ar,
            'blur_score': np.nan,
            'brightness_score': np.nan,
            'contrast_score': np.nan,
            'sharpness_score': np.nan,
            'noise_score': np.nan,
            'entropy_score': np.nan,
            'background_complexity': np.nan,
            'edge_density': np.nan,
            'image_quality_score': np.nan,
            'difficulty_score': 'Normal',
            'pretrain_id': pid,
            'sha256': row['sha256'],
            # Attach custom staging pipeline fields
            'phash': row['phash'],
            'train_only': row['train_only'],
            'field_eligible': row['field_eligible']
        }
        aligned_rows.append(aligned_row)

    df_aligned = pd.DataFrame(aligned_rows)
    print(f'✓ Aligned {len(df_aligned)} new rows to master dataset schema.')

    # 5. Concatenate with Existing Dataset
    df_merged = pd.concat([df_exist, df_aligned], ignore_index=True)

    print('\n=====================================================================')
    print('ROW COUNT RECONCILIATION')
    print('=====================================================================')
    print(f'Original Rows in dataset_final_training.csv : {len(df_exist):>8}')
    print(f'New QC-Passed Rows Added                     : {len(df_aligned):>8}')
    print(f'Total Merged Rows (dataset_v2_staging.csv)   : {len(df_merged):>8}')

    # 6. Save dataset_v2_staging.csv
    out_csv = 'ml_pipeline/dataset_v2_staging.csv'
    df_merged.to_csv(out_csv, index=False)
    df_merged.to_csv(os.path.join(OUTPUT_DIR, 'dataset_v2_staging.csv'), index=False)
    print(f'\n✓ Saved merged staging dataset to {out_csv}')

    # 7. Verification Checks
    print('\n=====================================================================')
    print('VERIFICATION CHECKS')
    print('=====================================================================')
    
    # Check 1: Old rows byte/value identity across all cells
    df_v2_read = pd.read_csv(out_csv, low_memory=False)
    old_subset = df_v2_read.iloc[:len(df_exist)][df_exist.columns]
    
    cell_mismatches = 0
    for col in df_exist.columns:
        c1 = df_exist[col]
        c2 = old_subset[col]
        both_null = c1.isnull() & c2.isnull()
        both_match = (c1 == c2) | both_null
        if not both_match.all():
            cell_mismatches += (~both_match).sum()

    old_pass = (cell_mismatches == 0)
    print(f'1. Old Rows Unchanged Check                : {"PASS (100% Value Identical - 0 cell mismatches)" if old_pass else "FAIL"}')
    
    # Check 2: New rows class_id validity
    valid_cids = set(head_classes.values())
    new_cids = set(df_aligned['class_id'])
    cids_valid = new_cids.issubset(valid_cids)
    print(f'2. New Rows Class ID Validity Check         : {"PASS (All IDs in 0..66)" if cids_valid else "FAIL"}')

    # Check 3: Critical column NaNs in new rows
    critical_cols = ['image_path', 'crop', 'disease', 'class_name', 'class_id', 'split', 'source_dataset', 'sha256']
    null_counts = df_aligned[critical_cols].isnull().sum().sum()
    print(f'3. Critical Columns Non-Null Check          : {"PASS (0 nulls)" if null_counts == 0 else f"FAIL ({null_counts} nulls)"}')

    # Check 4: Row count arithmetic
    count_pass = (len(df_merged) == len(df_exist) + len(df_aligned))
    print(f'4. Row Count Arithmetic Verification       : {"PASS (" + str(len(df_merged)) + " rows)" if count_pass else "FAIL"}')

if __name__ == '__main__':
    main()
