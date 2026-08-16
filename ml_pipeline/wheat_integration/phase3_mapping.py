import os, json
import pandas as pd

# Define output directory
OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGING_FILES = {
    'Bangladesh': 'ml_pipeline/staging_bangladesh.csv',
    'Long2023': 'ml_pipeline/staging_long2023.csv',
    'archive(1)': 'ml_pipeline/staging_archive1.csv',
    'CGIAR': 'ml_pipeline/staging_cgiar.csv'
}

# 15 existing wheat classes from class_map_final.json
EXISTING_15_WHEAT_CLASSES = [
    'Wheat_Aphid', 'Wheat_Black_Rust', 'Wheat_Blast', 'Wheat_Brown_Rust',
    'Wheat_Common_Root_Rot', 'Wheat_Fusarium_Head_Blight', 'Wheat_Healthy',
    'Wheat_Leaf_Blight', 'Wheat_Mildew', 'Wheat_Mite', 'Wheat_Septoria',
    'Wheat_Smut', 'Wheat_Stem_Fly', 'Wheat_Tan_Spot', 'Wheat_Yellow_Rust'
]

# Taxonomy mapping dictionary
TAXONOMY_MAP = {
    # Bangladesh
    'HealthyLeaf': 'Wheat_Healthy',
    'LeafBlight': 'Wheat_Leaf_Blight',
    'WheatBlast': 'Wheat_Blast',
    # Long 2023
    'BrownRust': 'Wheat_Brown_Rust',
    'Healthy': 'Wheat_Healthy',
    'Mildew': 'Wheat_Mildew',
    'Septoria': 'Wheat_Septoria',
    'YellowRust': 'Wheat_Yellow_Rust',
    # archive(1)
    'septoria': 'Wheat_Septoria',
    'stripe_rust': 'Wheat_Yellow_Rust',
    # CGIAR
    'healthy_wheat': 'Wheat_Healthy',
    'leaf_rust': 'Wheat_Brown_Rust',
    'stem_rust': 'Wheat_Black_Rust'
}

# Unmapped labels review list
UNMAPPED_LABELS = {
    'BlackPoint': {
        'source': 'Bangladesh',
        'raw_count': 303,
        'reason': 'Grain/kernel discoloration disease (not a foliage/root/rust/blast leaf disease in 15 target classes)',
        'recommendation': 'EXCLUDE (Park aside)'
    },
    'FusariumFootRot': {
        'source': 'Bangladesh',
        'raw_count': 250,
        'reason': 'Stem-base/seedling foot rot (differs from head blight ear infection and common root rot)',
        'recommendation': 'EXCLUDE (Park aside)'
    }
}

def main():
    print('=====================================================================')
    print('PHASE 3 -- TAXONOMY MAPPING EXECUTION')
    print('=====================================================================\n')

    all_staging_dfs = []
    mapping_records = []

    for source_name, filepath in STAGING_FILES.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Staging file missing: {filepath}')
            
        df = pd.read_csv(filepath)
        print(f'Loading {source_name:<12} ({filepath}): {len(df)} rows')
        
        # Apply taxonomy mapping
        wheat_classes = []
        for idx, row in df.iterrows():
            raw_lbl = row['raw_label']
            mapped_cls = TAXONOMY_MAP.get(raw_lbl, 'UNMAPPED')
            wheat_classes.append(mapped_cls)
            
            mapping_records.append({
                'source_dataset': source_name,
                'raw_label': raw_lbl,
                'wheat_class': mapped_cls
            })
            
        df['wheat_class'] = wheat_classes
        
        # Save updated staging CSV
        df.to_csv(filepath, index=False)
        print(f'✓ Updated {filepath} with wheat_class column ({len(df)} rows)')
        all_staging_dfs.append(df)

    # Save unmapped review CSV
    unmapped_rows = []
    for lbl, meta in UNMAPPED_LABELS.items():
        unmapped_rows.append({
            'raw_label': lbl,
            'source_dataset': meta['source'],
            'raw_count': meta['raw_count'],
            'reason': meta['reason'],
            'recommendation': meta['recommendation']
        })
    df_unmapped = pd.DataFrame(unmapped_rows)
    unmapped_csv_path = 'ml_pipeline/unmapped_review.csv'
    df_unmapped.to_csv(unmapped_csv_path, index=False)
    print(f'✓ Saved {unmapped_csv_path} with {len(df_unmapped)} unmapped label categories')

    # Also save inside wheat_integration directory
    df_unmapped.to_csv(os.path.join(OUTPUT_DIR, 'unmapped_review.csv'), index=False)

    print('\n=====================================================================')
    print('FULL TAXONOMY MAPPING TABLE (raw_label -> wheat_class)')
    print('=====================================================================')
    df_map = pd.DataFrame(mapping_records).drop_duplicates().sort_values(by=['source_dataset', 'raw_label'])
    print(f'{"Source Dataset":<15} | {"Raw Label":<20} | {"Mapped Wheat Class":<30}')
    print('-' * 70)
    for _, row in df_map.iterrows():
        print(f'{row["source_dataset"]:<15} | {row["raw_label"]:<20} | {row["wheat_class"]:<30}')
    print('-' * 70)

    print('\n=====================================================================')
    print('UNMAPPED LABELS REVIEW & DECISION TABLE (Parked Aside)')
    print('=====================================================================')
    print(f'{"Raw Label":<20} | {"Source":<12} | {"Count":<6} | {"Recommendation":<25}')
    print('-' * 70)
    for _, row in df_unmapped.iterrows():
        print(f'{row["raw_label"]:<20} | {row["source_dataset"]:<12} | {row["raw_count"]:<6} | {row["recommendation"]:<25}')
    print('-' * 70)

if __name__ == '__main__':
    main()
