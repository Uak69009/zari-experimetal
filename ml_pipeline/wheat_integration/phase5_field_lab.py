import os, json
import pandas as pd

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV = 'ml_pipeline/staging_deduplicated.csv'

FIELD_ELIGIBILITY_MAP = {
    "Bangladesh": "field",
    "CGIAR": "field"
}

JUSTIFICATIONS = {
    "Bangladesh": "In-situ field images collected from wheat fields in Bangladesh under natural sunlight and real crop canopy conditions.",
    "CGIAR": "Field survey images collected across wheat farms in Ethiopia under natural outdoor agricultural conditions."
}

def main():
    print('=====================================================================')
    print('PHASE 5 -- FIELD VS LAB CLASSIFICATION EXECUTION')
    print('=====================================================================\n')

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f'Input file missing: {INPUT_CSV}')

    df = pd.read_csv(INPUT_CSV)
    print(f'Loaded {len(df)} deduplicated rows from {INPUT_CSV}')

    # Save field_eligibility.json
    json_path_root = 'ml_pipeline/field_eligibility.json'
    json_path_out = os.path.join(OUTPUT_DIR, 'field_eligibility.json')
    
    with open(json_path_root, 'w') as f:
        json.dump(FIELD_ELIGIBILITY_MAP, f, indent=2)
    with open(json_path_out, 'w') as f:
        json.dump(FIELD_ELIGIBILITY_MAP, f, indent=2)
        
    print(f'✓ Saved {json_path_root}')

    # Determine field_eligible column
    # An image is field_eligible ONLY if:
    # 1) source_dataset is classified as 'field'
    # 2) train_only is False (e.g. resolution >= 200px short-side)
    field_eligible_flags = []
    for idx, row in df.iterrows():
        src = row['source_dataset']
        t_only = row['train_only']
        src_elig = FIELD_ELIGIBILITY_MAP.get(src, 'lab')
        
        is_eligible = (src_elig == 'field') and (not t_only)
        field_eligible_flags.append(is_eligible)

    df['field_eligible'] = field_eligible_flags

    # Save updated staging CSV
    df.to_csv(INPUT_CSV, index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, 'staging_deduplicated.csv'), index=False)
    print(f'✓ Updated {INPUT_CSV} with field_eligible column ({len(df)} rows)')

    print('\n=====================================================================')
    print('FIELD VS LAB CLASSIFICATION SUMMARY TABLE')
    print('=====================================================================')
    print(f'{"Dataset":<15} | {"Classification":<12} | {"Total Rows":<10} | {"Field-Eligible (Val/Test)":<25} | {"Train-Only":<12}')
    print('-' * 80)
    
    for ds in ['Bangladesh', 'CGIAR']:
        sub = df[df['source_dataset'] == ds]
        tot = len(sub)
        elig = sub['field_eligible'].sum()
        tr_only = tot - elig
        cls_name = FIELD_ELIGIBILITY_MAP[ds]
        print(f'{ds:<15} | {cls_name:<12} | {tot:<10} | {elig:<25} | {tr_only:<12}')
        
    print('-' * 80)

    print('\n=====================================================================')
    print('ONE-LINE JUSTIFICATIONS')
    print('=====================================================================')
    for ds, just in JUSTIFICATIONS.items():
        print(f'* {ds}: {just}')
    print('---------------------------------------------------------------------\n')

if __name__ == '__main__':
    main()
