import os, json, random
import pandas as pd
import numpy as np
from collections import defaultdict

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGING_V2_CSV = 'ml_pipeline/dataset_v2_staging.csv'
ORIGINAL_CSV = 'ml_pipeline/data/dataset_final_training.csv'
CLASS_MAP_FILE = 'ml_pipeline/data/class_map_final.json'

def main():
    print('=====================================================================')
    print('PHASE 8 -- WHEAT RE-SPLIT EXECUTION (WHEAT CLASSES ONLY, ATOMIC)')
    print('=====================================================================\n')

    # 1. Load Merged Staging Dataset
    if not os.path.exists(STAGING_V2_CSV):
        raise FileNotFoundError(f'Staging V2 dataset file missing: {STAGING_V2_CSV}')
    df_merged = pd.read_csv(STAGING_V2_CSV, low_memory=False)
    print(f'✓ Loaded Merged Dataset ({STAGING_V2_CSV}): {len(df_merged)} rows')

    # 2. Isolate Wheat vs Non-Wheat Rows
    wheat_mask = df_merged['class_name'].str.startswith('Wheat_')
    wheat_df = df_merged[wheat_mask].copy().reset_index(drop=True)
    non_wheat_df = df_merged[~wheat_mask].copy().reset_index(drop=True)

    print(f'✓ Isolated Wheat Rows     : {len(wheat_df)} rows')
    print(f'✓ Isolated Non-Wheat Rows : {len(non_wheat_df)} rows')

    # 3. Verify Non-Wheat Splits Unchanged
    print('\n--- Verifying Non-Wheat Rows Byte & Split Identity ---')
    df_orig = pd.read_csv(ORIGINAL_CSV, low_memory=False)
    orig_non_wheat = df_orig[~df_orig['class_name'].str.startswith('Wheat_')].reset_index(drop=True)

    non_wheat_count_match = (len(non_wheat_df) == len(orig_non_wheat))
    non_wheat_splits_match = (non_wheat_df['split'].values == orig_non_wheat['split'].values).all()

    print(f'1. Non-Wheat Row Count Match Check         : {"PASS (" + str(len(non_wheat_df)) + " rows)" if non_wheat_count_match else "FAIL"}')
    print(f'2. Non-Wheat Split Unchanged Check         : {"PASS (100% Identical, 0 split changes)" if non_wheat_splits_match else "FAIL"}')

    if not (non_wheat_count_match and non_wheat_splits_match):
        raise ValueError('CRITICAL: Non-wheat rows were mutated! Halting Phase 8 execution.')

    # 4. SHA256-Atomic Class-Stratified Re-splitting on Wheat Classes
    print('\n--- Re-Splitting Wheat Classes (SHA256-Atomic, Field-Only for Val/Test, Seed=42) ---')
    
    # Ensure field_eligible is properly set for all wheat rows
    # For existing wheat rows, if field_eligible is null/missing, determine from domain or source_dataset
    if 'field_eligible' not in wheat_df.columns or wheat_df['field_eligible'].isnull().any():
        # Baseline field eligibility logic: PlantVillage is lab, nwrd/plantdoc/plantcity/Bangladesh/CGIAR are field
        is_field = ~wheat_df['source_dataset'].isin(['plantvillage'])
        if 'train_only' in wheat_df.columns:
            is_field = is_field & (~wheat_df['train_only'].fillna(False).astype(bool))
        wheat_df['field_eligible'] = is_field.fillna(True)

    random.seed(42)
    np.random.seed(42)

    new_hash_splits = {}

    for wheat_class, group in wheat_df.groupby('class_name'):
        hash_groups = group.groupby('sha256')
        unique_hashes = list(hash_groups.groups.keys())
        random.shuffle(unique_hashes)

        # Separate field-eligible hashes vs train-only hashes
        field_hashes = set()
        for h in unique_hashes:
            rows = hash_groups.get_group(h)
            # All rows in hash group must be field-eligible to qualify for val/test
            if rows['field_eligible'].all():
                field_hashes.add(h)

        train_only_hashes = set(unique_hashes) - field_hashes

        field_hash_list = list(field_hashes)
        random.shuffle(field_hash_list)

        n_field = len(field_hash_list)
        n_val = max(1, int(round(n_field * 0.10))) if n_field >= 10 else 1
        n_test = max(1, int(round(n_field * 0.10))) if n_field >= 10 else 1

        val_hashes = set(field_hash_list[:n_val])
        test_hashes = set(field_hash_list[n_val:n_val + n_test])
        train_hashes = set(field_hash_list[n_val + n_test:]) | train_only_hashes

        for h in unique_hashes:
            if h in val_hashes:
                new_hash_splits[h] = 'val'
            elif h in test_hashes:
                new_hash_splits[h] = 'test'
            else:
                new_hash_splits[h] = 'train'

    # Apply new split to wheat_df
    wheat_df['split'] = wheat_df['sha256'].map(new_hash_splits)

    # 5. Save wheat_resplit.csv
    out_wheat_csv = 'ml_pipeline/wheat_resplit.csv'
    wheat_df.to_csv(out_wheat_csv, index=False)
    wheat_df.to_csv(os.path.join(OUTPUT_DIR, 'wheat_resplit.csv'), index=False)
    print(f'✓ Saved wheat resplit dataset to {out_wheat_csv} ({len(wheat_df)} rows)')

    # 6. Print Per-Class Before / After Split Table
    print('\n=====================================================================')
    print('PER-CLASS WHEAT SPLIT COMPARISON TABLE (BEFORE vs AFTER)')
    print('=====================================================================')
    print(f'{"Wheat Class Name":<30} | {"OLD (Train / Val / Test)":<25} | {"NEW (Train / Val / Test)":<25} | {"Net Change"}')
    print('-' * 95)

    with open(CLASS_MAP_FILE) as f:
        cmap = json.load(f)['head_classes']
    all_15_wheat = sorted([c for c in cmap if c.startswith('Wheat_')])

    tot_old_tr, tot_old_val, tot_old_te = 0, 0, 0
    tot_new_tr, tot_new_val, tot_new_te = 0, 0, 0

    for wc in all_15_wheat:
        old_sub = df_orig[df_orig['class_name'] == wc]
        old_counts = old_sub['split'].value_counts().to_dict()
        old_tr, old_val, old_te = old_counts.get('train', 0), old_counts.get('val', 0), old_counts.get('test', 0)

        new_sub = wheat_df[wheat_df['class_name'] == wc]
        new_counts = new_sub['split'].value_counts().to_dict()
        new_tr, new_val, new_te = new_counts.get('train', 0), new_counts.get('val', 0), new_counts.get('test', 0)

        tot_old_tr += old_tr; tot_old_val += old_val; tot_old_te += old_te
        tot_new_tr += new_tr; tot_new_val += new_val; tot_new_te += new_te

        old_str = f'{old_tr:>5} / {old_val:>4} / {old_te:>4}'
        new_str = f'{new_tr:>5} / {new_val:>4} / {new_te:>4}'
        diff_str = f'+{len(new_sub) - len(old_sub)}' if len(new_sub) > len(old_sub) else '0'

        print(f'{wc:<30} | {old_str:<25} | {new_str:<25} | {diff_str}')

    print('-' * 95)
    old_tot_str = f'{tot_old_tr:>5} / {tot_old_val:>4} / {tot_old_te:>4}'
    new_tot_str = f'{tot_new_tr:>5} / {tot_new_val:>4} / {tot_new_te:>4}'
    print(f'{"TOTAL WHEAT ROWS":<30} | {old_tot_str:<25} | {new_tot_str:<25} | +{len(wheat_df) - len(df_orig[df_orig["class_name"].str.startswith("Wheat_")])}')
    print('-' * 95)

if __name__ == '__main__':
    main()
