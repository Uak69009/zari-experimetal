import os, json
import pandas as pd
import numpy as np

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGING_V2_CSV = 'ml_pipeline/dataset_v2_staging.csv'
WHEAT_RESPLIT_CSV = 'ml_pipeline/wheat_resplit.csv'
ORIGINAL_CSV = 'ml_pipeline/data/dataset_final_training.csv'
FINAL_V2_CSV = 'ml_pipeline/data/dataset_final_training_v2.csv'
CLASS_MAP_FILE = 'ml_pipeline/data/class_map_final.json'

def main():
    print('=====================================================================')
    print('PHASE 9 -- RECOMBINE EXECUTION')
    print('=====================================================================\n')

    # 1. Load Components
    if not os.path.exists(STAGING_V2_CSV):
        raise FileNotFoundError(f'Missing staging file: {STAGING_V2_CSV}')
    if not os.path.exists(WHEAT_RESPLIT_CSV):
        raise FileNotFoundError(f'Missing wheat resplit file: {WHEAT_RESPLIT_CSV}')
        
    df_staging = pd.read_csv(STAGING_V2_CSV, low_memory=False)
    df_wheat_new = pd.read_csv(WHEAT_RESPLIT_CSV, low_memory=False)
    
    print(f'✓ Loaded Staging V2 Dataset ({STAGING_V2_CSV}): {len(df_staging)} rows')
    print(f'✓ Loaded Wheat Re-Split Dataset ({WHEAT_RESPLIT_CSV}): {len(df_wheat_new)} rows')

    # 2. Isolate Non-Wheat Rows from Staging
    non_wheat_mask = ~df_staging['class_name'].str.startswith('Wheat_')
    df_non_wheat = df_staging[non_wheat_mask].copy().reset_index(drop=True)
    print(f'✓ Isolated Non-Wheat Rows: {len(df_non_wheat)} rows')

    # 3. Recombine Non-Wheat + Re-Split Wheat
    df_final_v2 = pd.concat([df_non_wheat, df_wheat_new], ignore_index=True)
    print(f'✓ Recombined Final Dataset: {len(df_final_v2)} rows (Expected: 124,321)')

    # 4. Comprehensive Verification Checks
    print('\n=====================================================================')
    print('VERIFICATION CHECKS & CHECKSUM PROOF')
    print('=====================================================================')

    # Check 1: Total Row Count Arithmetic
    assert len(df_final_v2) == 124321, f"Expected 124321, got {len(df_final_v2)}"
    print(f'1. Total Row Count Check                   : PASS ({len(df_final_v2)} rows)')

    # Check 2: Non-Wheat Split & Cell Value Identity against original dataset_final_training.csv
    df_orig = pd.read_csv(ORIGINAL_CSV, low_memory=False)
    orig_non_wheat = df_orig[~df_orig['class_name'].str.startswith('Wheat_')].reset_index(drop=True)
    
    mismatches = 0
    for col in df_orig.columns:
        c1 = orig_non_wheat[col]
        c2 = df_non_wheat[col]
        both_null = c1.isnull() & c2.isnull()
        both_match = (c1 == c2) | both_null
        if not both_match.all():
            mismatches += (~both_match).sum()

    assert mismatches == 0, f"Non-wheat rows mutated! Cell mismatches: {mismatches}"
    print(f'2. Non-Wheat Rows Byte/Value Identity Check: PASS (100% Identical, 0 cell mismatches)')

    # Check 3: Class ID Validity
    with open(CLASS_MAP_FILE) as f:
        cmap = json.load(f)['head_classes']
    head_cids = set(cmap.values())
    valid_cids = head_cids.union({-1})
    
    actual_cids = set(df_final_v2['class_id'].dropna().astype(int))
    assert actual_cids.issubset(valid_cids), f"Invalid class_id found: {actual_cids - valid_cids}"
    
    wheat_v2 = df_final_v2[df_final_v2['class_name'].str.startswith('Wheat_')]
    wheat_cids_actual = set(wheat_v2['class_id'].astype(int))
    expected_wheat_cids = set(range(52, 67))
    assert wheat_cids_actual == expected_wheat_cids, f"Wheat class_ids mismatch: {wheat_cids_actual}"
    print(f'3. Class ID Validity Check                 : PASS (Master IDs in -1..66, Wheat IDs exact 52..66)')

    # Check 4: Zero Split Leakage Check (every hash belongs to exactly one split)
    wheat_hash_splits = wheat_v2.groupby('sha256')['split'].nunique()
    leaked_hashes = (wheat_hash_splits > 1).sum()
    assert leaked_hashes == 0, f"Hash split leakage detected: {leaked_hashes} hashes in multiple splits!"
    print(f'4. Zero SHA256 Split Leakage Check         : PASS (100% Atomic: 0 hashes leaked across splits)')

    # 5. Save dataset_final_training_v2.csv
    df_final_v2.to_csv(FINAL_V2_CSV, index=False)
    df_final_v2.to_csv(os.path.join(OUTPUT_DIR, 'dataset_final_training_v2.csv'), index=False)
    print(f'\n✅ Saved NEW dataset to: {FINAL_V2_CSV} ({len(df_final_v2)} rows)')
    print(f'✅ Original dataset untouched: {ORIGINAL_CSV} ({len(df_orig)} rows)')

    # 6. Overall Split & Wheat Class Distributions
    print('\n=====================================================================')
    print('DATASET V2 FINAL SPLIT DISTRIBUTION (OVERALL & WHEAT)')
    print('=====================================================================')
    print('OVERALL DATASET V2 SPLIT DISTRIBUTION (124,321 rows):')
    print(df_final_v2['split'].value_counts())

    print('\nWHEAT-ONLY SPLIT DISTRIBUTION (15,171 rows):')
    print(wheat_v2['split'].value_counts())

    print('\nWHEAT CLASS ROW COUNTS (dataset_final_training_v2.csv):')
    print(wheat_v2['class_name'].value_counts().to_string())

if __name__ == '__main__':
    main()
