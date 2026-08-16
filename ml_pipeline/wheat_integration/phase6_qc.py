import os, json, cv2
import pandas as pd
import numpy as np

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV = 'ml_pipeline/staging_deduplicated.csv'

def main():
    print('=====================================================================')
    print('PHASE 6 -- QUALITY CONTROL EXECUTION')
    print('=====================================================================\n')

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f'Input file missing: {INPUT_CSV}')

    df = pd.read_csv(INPUT_CSV)
    print(f'Loaded {len(df)} deduplicated rows from {INPUT_CSV}')

    # STEP 1 & 4: Image Quality & Blur Audit
    blur_scores = []
    dropped_rows = []
    keep_indices = []

    print('\n--- STEP 1 & 4: IMAGE QUALITY & BLUR AUDIT ---')
    for idx, row in df.iterrows():
        fp = row['filepath']
        try:
            img = cv2.imread(fp)
            if img is None:
                dropped_rows.append((row, 'Unreadable/Corrupt file'))
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(blur_val)
            
            # Threshold for extreme blur: Laplacian variance < 5.0
            if blur_val < 5.0:
                dropped_rows.append((row, f'Extreme blur (Laplacian variance={blur_val:.2f} < 5.0)'))
            else:
                keep_indices.append(idx)
        except Exception as e:
            dropped_rows.append((row, f'Processing error: {str(e)}'))

    print(f'Total images audited: {len(df)}')
    print(f'Total unusable/extreme-blur images dropped: {len(dropped_rows)}')
    if dropped_rows:
        print('Dropped image details:')
        for r, reason in dropped_rows:
            print(f'  - [{r["source_dataset"]} | {r["wheat_class"]} | {os.path.basename(r["filepath"])}]: {reason}')

    df_qc = df.iloc[keep_indices].reset_index(drop=True)
    print(f'Surviving images post-QC: {len(df_qc)}\n')

    # STEP 2: Source Concentration Risk Analysis
    print('=====================================================================')
    print('SOURCE CONCENTRATION ANALYSIS PER WHEAT CLASS')
    print('=====================================================================')
    
    cmap_file = 'ml_pipeline/data/class_map_final.json'
    with open(cmap_file) as f:
        cmap = json.load(f)['head_classes']
    all_15_wheat = sorted([c for c in cmap if c.startswith('Wheat_')])
    
    concentration_rows = []
    for wc in all_15_wheat:
        sub = df_qc[df_qc['wheat_class'] == wc]
        tot_new = len(sub)
        if tot_new == 0:
            concentration_rows.append({
                'wheat_class': wc,
                'total_new': 0,
                'sources': 'None',
                'max_conc_pct': 0.0,
                'status': 'NO NEW DATA'
            })
        else:
            vc = sub['source_dataset'].value_counts()
            top_src = vc.index[0]
            top_cnt = vc.iloc[0]
            pct = (top_cnt / tot_new) * 100.0
            
            src_str = ', '.join([f'{s}: {c} ({c/tot_new*100:.1f}%)' for s, c in vc.items()])
            status = 'FLAGGED (>90% single-source risk)' if pct > 90.0 else 'PASSED'
            
            concentration_rows.append({
                'wheat_class': wc,
                'total_new': tot_new,
                'sources': src_str,
                'max_conc_pct': round(pct, 2),
                'status': status
            })

    print(f'{"Wheat Class":<30} | {"New Images":<10} | {"Max Source %":<14} | {"Status & Source Breakdown":<45}')
    print('-' * 105)
    for r in concentration_rows:
        print(f'{r["wheat_class"]:<30} | {r["total_new"]:<10} | {r["max_conc_pct"]:>12.2f}% | {r["status"]:<35} ({r["sources"]})')
    print('-' * 105)

    # STEP 5: Save staging_qc_passed.csv
    out_csv = 'ml_pipeline/staging_qc_passed.csv'
    df_qc.to_csv(out_csv, index=False)
    df_qc.to_csv(os.path.join(OUTPUT_DIR, 'staging_qc_passed.csv'), index=False)
    print(f'\n✓ Saved QC passed dataset to {out_csv} ({len(df_qc)} rows)')

    print('\n=====================================================================')
    print('FINAL QC PASSED DATASET COMPOSITION BY CLASS')
    print('=====================================================================')
    print(df_qc.groupby(['wheat_class', 'source_dataset', 'field_eligible']).size().reset_index(name='count'))

if __name__ == '__main__':
    main()
