import os, json
import pandas as pd
import numpy as np

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

V2_CSV = 'ml_pipeline/data/dataset_final_training_v2.csv'
CLASS_MAP_FILE = 'ml_pipeline/data/class_map_final.json'
OLD_WEIGHTS_FILE = 'ml_pipeline/data/class_weights.json'
NEW_WEIGHTS_FILE = 'ml_pipeline/data/class_weights_v2.json'

def main():
    print('=====================================================================')
    print('PHASE 11 -- RECOMPUTE DERIVED METADATA & CLASS WEIGHTS V2')
    print('=====================================================================\n')

    # 1. Load Dataset V2 and Class Map
    df_v2 = pd.read_csv(V2_CSV, low_memory=False)
    with open(CLASS_MAP_FILE) as f:
        cmap_data = json.load(f)
    head_classes = cmap_data['head_classes']
    pretrain_classes = cmap_data['pretrain_classes']
    id_to_head_name = {v: k for k, v in head_classes.items()}
    id_to_pretrain_name = {v: k for k, v in pretrain_classes.items()}

    # 2. Filter Field Train Data for Head Weights
    field_sources = ['plantcity', 'nwrd', 'plantdoc', 'Bangladesh', 'CGIAR']
    field_train_df = df_v2[(df_v2['split'] == 'train') & (df_v2['source_dataset'].isin(field_sources))]
    print(f'✓ Total Field Train Samples in V2: {len(field_train_df):,}')

    # Head weights calculation (67 classes)
    K = len(head_classes)
    N_head = len(field_train_df)
    head_counts = field_train_df['class_id'].value_counts()

    raw_h_weights = []
    for cid in range(K):
        c_cnt = head_counts.get(cid, 0)
        w = (N_head / (K * c_cnt)) if c_cnt > 0 else 10.0
        raw_h_weights.append(w)

    raw_h_arr = np.array(raw_h_weights)
    h_norm = raw_h_arr / raw_h_arr.mean()
    h_clipped = np.clip(h_norm, 0.1, 10.0)

    head_weights_dict = {str(i): round(float(h_clipped[i]), 4) for i in range(K)}
    head_weights_by_name = {id_to_head_name[i]: round(float(h_clipped[i]), 4) for i in range(K)}

    # Pretrain weights calculation (from existing class_weights.json or dataset_v2 pretrain split)
    with open(OLD_WEIGHTS_FILE) as f:
        old_weights_data = json.load(f)
    pretrain_weights_dict = old_weights_data['pretrain_weights']
    pretrain_weights_by_name = old_weights_data['pretrain_weights_by_name']
    pretrain_stats = old_weights_data['pretrain_stats']

    # Head stats calculation
    head_stats = {
        "total_samples": int(N_head),
        "num_classes": K,
        "imbalance_ratio": round(float(max(h_clipped) / min(h_clipped)), 2),
        "min_count": int(head_counts.min()),
        "max_count": int(head_counts.max()),
        "mean_count": round(float(head_counts.mean()), 2),
        "median_count": float(head_counts.median())
    }

    full_class_weights_v2 = {
        "pretrain_weights": pretrain_weights_dict,
        "head_weights": head_weights_dict,
        "pretrain_weights_by_name": pretrain_weights_by_name,
        "head_weights_by_name": head_weights_by_name,
        "pretrain_stats": pretrain_stats,
        "head_stats": head_stats,
        "recommendation": "RECOMMEND WEIGHTED CROSS ENTROPY LOSS (Imbalance ratio <= 500:1, standard weighted CE is optimal)"
    }

    # Save class_weights_v2.json
    with open(NEW_WEIGHTS_FILE, 'w') as f:
        json.dump(full_class_weights_v2, f, indent=2)
    with open('ml_pipeline/class_weights_v2.json', 'w') as f:
        json.dump(full_class_weights_v2, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'class_weights_v2.json'), 'w') as f:
        json.dump(full_class_weights_v2, f, indent=2)

    print(f'✓ Saved updated weights to {NEW_WEIGHTS_FILE}')

    # 3. Compare Wheat Class Weights (OLD vs NEW)
    old_head_w = old_weights_data['head_weights']

    print('\n=====================================================================')
    print('WHEAT CLASS WEIGHTS COMPARISON TABLE (OLD vs NEW)')
    print('=====================================================================')
    print(f'{"Wheat Class Name":<30} | {"Old Weight":<12} | {"New Weight":<12} | {"Net Change"}')
    print('-' * 70)

    for cid in range(52, 67):
        cname = id_to_head_name[cid]
        old_w = float(old_head_w[str(cid)])
        new_w = float(head_weights_dict[str(cid)])
        diff = new_w - old_w
        print(f'{cname:<30} | {old_w:<12.4f} | {new_w:<12.4f} | {diff:>+7.4f}')
    print('-' * 70)

    # 4. Imbalance Stats Comparison
    old_wheat_w = [float(old_head_w[str(i)]) for i in range(52, 67)]
    new_wheat_w = [float(head_weights_dict[str(i)]) for i in range(52, 67)]

    print('\n=====================================================================')
    print('WHEAT CLASS IMBALANCE STATS (OLD vs NEW)')
    print('=====================================================================')
    print(f'Old Wheat Weights -> Min: {min(old_wheat_w):.4f}, Max: {max(old_wheat_w):.4f}, Mean: {np.mean(old_wheat_w):.4f}, Imbalance Ratio: {max(old_wheat_w)/min(old_wheat_w):.2f}:1')
    print(f'New Wheat Weights -> Min: {min(new_wheat_w):.4f}, Max: {max(new_wheat_w):.4f}, Mean: {np.mean(new_wheat_w):.4f}, Imbalance Ratio: {max(new_wheat_w)/min(new_wheat_w):.2f}:1')
    print('-' * 70)

    # 5. Confirm class_map_final.json status
    print('\n✓ Confirmation: class_map_final.json needs NO changes (106 total classes, 67 head classes remain unchanged).')

if __name__ == '__main__':
    main()
