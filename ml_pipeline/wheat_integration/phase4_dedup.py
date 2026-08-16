import os, json, hashlib, time
import pandas as pd
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGING_FILES = {
    'Bangladesh': 'ml_pipeline/staging_bangladesh.csv',
    'Long2023': 'ml_pipeline/staging_long2023.csv',
    'archive(1)': 'ml_pipeline/staging_archive1.csv',
    'CGIAR': 'ml_pipeline/staging_cgiar.csv'
}

EXISTING_CSV = 'ml_pipeline/data/dataset_final_training.csv'
CLASS_MAP_FILE = 'ml_pipeline/data/class_map_final.json'

def compute_phash_single(path):
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None

def main():
    print('=====================================================================')
    print('PHASE 4 -- DEDUPLICATION EXECUTION')
    print('=====================================================================\n')

    # STEP 1: Load staging CSVs
    dfs = []
    start_total = 0
    print('--- STEP 1: LOAD ALL 4 STAGING CSVs ---')
    for source_name, filepath in STAGING_FILES.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Staging file missing: {filepath}')
        df = pd.read_csv(filepath)
        print(f'{source_name:<12}: {len(df)} rows')
        dfs.append(df)
        start_total += len(df)
    print(f'Total Starting New Scoped Rows: {start_total}\n')

    # STEP 2: Within-Dataset Exact Dedup
    print('--- STEP 2: WITHIN-DATASET EXACT DEDUP (drop duplicate sha256 within source) ---')
    within_dedup_dfs = []
    within_exact_dropped = 0
    for df in dfs:
        source_name = df['source_dataset'].iloc[0]
        before = len(df)
        df_clean = df.drop_duplicates(subset=['sha256'], keep='first').copy()
        after = len(df_clean)
        dropped = before - after
        within_exact_dropped += dropped
        print(f'{source_name:<12}: {before} → {after} (removed {dropped})')
        within_dedup_dfs.append(df_clean)
        
    combined_within = pd.concat(within_dedup_dfs, ignore_index=True)
    print(f'Total after Within-Dataset Exact Dedup: {len(combined_within)} (Total Removed: {within_exact_dropped})\n')

    # STEP 3: Cross-Dataset Exact Dedup Against Existing Corpus
    print('--- STEP 3: CROSS-DATASET EXACT DEDUP AGAINST EXISTING CORPUS ---')
    df_exist = pd.read_csv(EXISTING_CSV)
    existing_hashes = set(df_exist['sha256'].dropna())
    print(f'Loaded {len(existing_hashes)} unique SHA256 hashes from existing {EXISTING_CSV}')
    
    combined_within['exists_in_original'] = combined_within['sha256'].isin(existing_hashes)
    cross_exact_dupes = combined_within[combined_within['exists_in_original'] == True]
    clean_cross_exact = combined_within[combined_within['exists_in_original'] == False].drop(columns=['exists_in_original']).reset_index(drop=True)
    
    cross_exact_dropped = len(cross_exact_dupes)
    print(f'Cross-dataset exact duplicates removed: {cross_exact_dropped}')
    if cross_exact_dropped > 0:
        print('Breakdown of cross-dataset exact duplicates removed:')
        print(cross_exact_dupes[['source_dataset', 'raw_label', 'wheat_class']].value_counts())
    print(f'Total after Cross-Dataset Exact Dedup: {len(clean_cross_exact)}\n')

    # STEP 4: Within-New Data Perceptual Near-Duplicate Check
    print('--- STEP 4: WITHIN-NEW DATA PERCEPTUAL NEAR-DUPLICATE CHECK (Hamming dist <= 5) ---')
    phashes_new = [imagehash.hex_to_hash(ph) for ph in clean_cross_exact['phash']]
    near_dupe_pairs_within = []
    drop_indices_within = set()

    for i, j in combinations(range(len(clean_cross_exact)), 2):
        dist = phashes_new[i] - phashes_new[j]
        if dist <= 5:
            near_dupe_pairs_within.append((i, j, dist))
            drop_indices_within.add(j)

    within_perceptual_dropped = len(drop_indices_within)
    print(f'Within-new perceptual near-duplicate pairs found: {len(near_dupe_pairs_within)}')
    print(f'Unique images dropped due to within-new perceptual near-dupes: {within_perceptual_dropped}')
    if near_dupe_pairs_within:
        print('Sample 5 near-duplicate pairs within new data:')
        for i, j, dist in near_dupe_pairs_within[:5]:
            r_i = clean_cross_exact.iloc[i]
            r_j = clean_cross_exact.iloc[j]
            print(f'  - Dist={dist}: [{r_i["source_dataset"]} | {r_i["wheat_class"]}] vs [{r_j["source_dataset"]} | {r_j["wheat_class"]}]')

    clean_within_perceptual = clean_cross_exact.drop(index=list(drop_indices_within)).reset_index(drop=True)
    print(f'Total after Within-New Perceptual Dedup: {len(clean_within_perceptual)}\n')

    # STEP 5: Cross-Dataset Perceptual Check Against Existing Wheat Images
    print('--- STEP 5: CROSS-DATASET PERCEPTUAL CHECK AGAINST EXISTING WHEAT CORPUS ---')
    with open(CLASS_MAP_FILE) as f:
        cmap = json.load(f)['head_classes']
    wheat_cids = set([cmap[c] for c in cmap if c.startswith('Wheat_')])
    exist_wheat = df_exist[df_exist['class_id'].isin(wheat_cids)].copy()
    exist_wheat_paths = exist_wheat['image_path'].tolist()
    print(f'Existing wheat corpus images: {len(exist_wheat_paths)}')

    print('Computing perceptual hashes for existing wheat images...')
    t0 = time.time()
    with ProcessPoolExecutor() as executor:
        exist_phashes_raw = list(executor.map(compute_phash_single, exist_wheat_paths, chunksize=500))
    exist_phashes = [imagehash.hex_to_hash(ph) for ph in exist_phashes_raw if ph is not None]
    print(f'Computed {len(exist_phashes)} existing wheat phashes in {time.time() - t0:.2f}s')

    drop_indices_cross_perceptual = set()
    near_dupe_pairs_cross = []
    
    new_phashes_list = [imagehash.hex_to_hash(ph) for ph in clean_within_perceptual['phash']]

    for idx, new_ph in enumerate(new_phashes_list):
        for ex_ph in exist_phashes:
            dist = new_ph - ex_ph
            if dist <= 5:
                near_dupe_pairs_cross.append((idx, dist))
                drop_indices_cross_perceptual.add(idx)
                break

    cross_perceptual_dropped = len(drop_indices_cross_perceptual)
    print(f'Cross-dataset perceptual near-duplicates found against existing corpus: {cross_perceptual_dropped}')
    if near_dupe_pairs_cross:
        print('Sample 5 cross-dataset perceptual near-duplicate pairs:')
        for idx, dist in near_dupe_pairs_cross[:5]:
            r = clean_within_perceptual.iloc[idx]
            print(f'  - Dist={dist}: [{r["source_dataset"]} | {r["wheat_class"]} | {os.path.basename(r["filepath"])}]')

    final_dedup_df = clean_within_perceptual.drop(index=list(drop_indices_cross_perceptual)).reset_index(drop=True)
    print(f'\nFinal Remaining Deduplicated New Wheat Images: {len(final_dedup_df)}\n')

    # STEP 6: Save staging_deduplicated.csv
    out_csv = 'ml_pipeline/staging_deduplicated.csv'
    final_dedup_df.to_csv(out_csv, index=False)
    final_dedup_df.to_csv(os.path.join(OUTPUT_DIR, 'staging_deduplicated.csv'), index=False)
    print(f'✓ Saved deduplicated dataset to {out_csv} ({len(final_dedup_df)} rows)')

    # STEP 7: Print Full Dedup Summary Table
    print('\n=====================================================================')
    print('FULL DEDUPLICATION SUMMARY TABLE')
    print('=====================================================================')
    summary_data = [
        ('Starting Scoped Input Rows', start_total),
        ('Within-dataset Exact Duplicates Removed', within_exact_dropped),
        ('Cross-dataset Exact Duplicates Removed (vs dataset_final_training.csv)', cross_exact_dropped),
        ('Within-new Perceptual Near-duplicates Removed (Hamming dist <= 5)', within_perceptual_dropped),
        ('Cross-dataset Perceptual Near-duplicates Removed (vs existing wheat corpus)', cross_perceptual_dropped),
        ('FINAL REMAINING UNIQUE NEW WHEAT IMAGES', len(final_dedup_df))
    ]
    print(f'{"Deduplication Stage / Category":<60} | {"Count":<10}')
    print('-' * 73)
    for category, count in summary_data:
        print(f'{category:<60} | {count:<10}')
    print('-' * 73)

if __name__ == '__main__':
    main()
