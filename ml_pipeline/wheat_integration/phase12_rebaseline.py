import os, json, time, re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, roc_auc_score

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
os.makedirs(OUTPUT_DIR, exist_ok=True)

V2_CSV = 'ml_pipeline/data/dataset_final_training_v2.csv'
MODEL_PATH = 'ml_pipeline/models/phase2_edl_model.pth'
CLASS_MAP_FILE = 'ml_pipeline/data/class_map_final.json'
REPORT_FILE = 'ml_pipeline/rebaseline_report.txt'

NUM_CLASSES = 67

class FieldTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df['image_path'].astype(str).tolist()
        self.labels = self.df['class_id'].astype(int).tolist()
        self.class_names = self.df['class_name'].astype(str).tolist()
        
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path_str = self.image_paths[idx]
        lbl = self.labels[idx]
        cname = self.class_names[idx]
        
        img = Image.open(path_str).convert('RGB')
        tensor_img = self.transform(img)
        return tensor_img, lbl, cname

def main():
    print('=====================================================================')
    print('PHASE 12 -- RE-BASELINE EXECUTION (CURRENT MODEL ON NEW TEST SET)')
    print('=====================================================================\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using compute device: {device}')
    if device.type == 'cuda':
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')

    # 1. Load Class Map
    with open(CLASS_MAP_FILE) as f:
        head_classes = json.load(f)['head_classes']
    id_to_name = {v: k for k, v in head_classes.items()}

    # 2. Load Model
    print('\n[STEP 1] Loading Current Production Model (phase2_edl_model.pth)...')
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model checkpoint missing: {MODEL_PATH}')

    model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False)
    model.reset_classifier(0)
    model.classifier = nn.Linear(1280, NUM_CLASSES)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    print(f'✓ Successfully loaded baseline model from {MODEL_PATH}')

    # 3. Load New Test Set
    print('\n[STEP 2] Loading New Test Set (dataset_final_training_v2.csv)...')
    df_v2 = pd.read_csv(V2_CSV, low_memory=False)
    test_df = df_v2[(df_v2['split'] == 'test') & (df_v2['class_id'] >= 0)].copy().reset_index(drop=True)
    print(f'✓ New Test Set Loaded: {len(test_df):,} field test images')

    test_dataset = FieldTestDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64 if device.type == 'cuda' else 16,
        shuffle=False,
        num_workers=8,
        pin_memory=(device.type == 'cuda')
    )

    # 4. Evaluation Loop
    print('\n[STEP 3] Evaluating Baseline Model on New Test Set...')
    all_targets = []
    all_preds = []
    all_probs = []
    all_uncertainties = []

    t0 = time.time()
    with torch.no_grad():
        for images, labels, cnames in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                uncertainties = float(NUM_CLASSES) / S.squeeze(-1)
                preds = probs.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_uncertainties.extend(uncertainties.cpu().numpy())

    eval_time = time.time() - t0
    print(f'✓ Inference complete in {eval_time:.2f}s ({len(test_df)/eval_time:.1f} img/s)')

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_uncertainties = np.array(all_uncertainties)

    # Metrics computation
    overall_acc = accuracy_score(all_targets, all_preds) * 100.0
    mean_unc = np.mean(all_uncertainties)
    
    # Compute One-vs-Rest AUROC
    try:
        auroc_score = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except Exception:
        auroc_score = np.nan

    print(f'\n✓ Overall Test Accuracy on New Test Set : {overall_acc:.2f}%')
    print(f'✓ Overall AUROC Score                   : {auroc_score:.4f}')
    print(f'✓ Mean Uncertainty                     : {mean_unc:.4f}')

    # Per-class accuracy computation
    per_class_acc = {}
    for cid in range(NUM_CLASSES):
        mask = (all_targets == cid)
        if mask.sum() > 0:
            c_acc = (all_preds[mask] == cid).mean() * 100.0
            per_class_acc[cid] = c_acc
        else:
            per_class_acc[cid] = np.nan

    # Old baseline test metrics (from original test evaluation on 6,648 images)
    old_metrics = {
        'overall': 97.32,
        'Wheat_Tan_Spot': 66.27,
        'Wheat_Leaf_Blight': 69.32,
        'Wheat_Black_Rust': 77.14,
        'Wheat_Septoria': 89.08,
        'Wheat_Brown_Rust': 87.05,
        'Wheat_Blast': 71.43,
        'Wheat_Healthy': 93.46,
        'Wheat_Yellow_Rust': 97.84,
        'Wheat_Aphid': 94.23,
        'Wheat_Common_Root_Rot': 84.72,
        'Wheat_Fusarium_Head_Blight': 83.33,
        'Wheat_Mildew': 94.83,
        'Wheat_Mite': 95.35,
        'Wheat_Smut': 97.08,
        'Wheat_Stem_Fly': 86.21
    }

    # 5. Build Re-baseline Report Output
    report_lines = []
    report_lines.append("=====================================================================")
    report_lines.append("ZARI.ai -- PHASE 12 RE-BASELINE EVALUATION REPORT")
    report_lines.append("=====================================================================")
    report_lines.append(f"Date / Timestamp           : {time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    report_lines.append(f"Evaluated Model Checkpoint : {MODEL_PATH}")
    report_lines.append(f"Evaluation Test Set        : {V2_CSV} (split == 'test')")
    report_lines.append(f"Total Test Field Images    : {len(test_df):,} images")
    report_lines.append(f"Overall Accuracy           : {overall_acc:.2f}%")
    report_lines.append(f"Overall AUROC              : {auroc_score:.4f}")
    report_lines.append(f"Mean Uncertainty           : {mean_unc:.4f}\n")

    report_lines.append("=====================================================================")
    report_lines.append("COMPARISON TABLE: OLD TEST SET vs. NEW RE-BASELINE TEST SET")
    report_lines.append("=====================================================================")
    report_lines.append(f'{"Metric / Wheat Class":<30} | {"Old Test Set (6,648)":<22} | {"New Baseline (6,709)":<22} | {"Baseline Delta"}')
    report_lines.append('-' * 90)

    report_lines.append(f'{"Overall Test Accuracy":<30} | {old_metrics["overall"]:>20.2f}% | {overall_acc:>20.2f}% | {overall_acc - old_metrics["overall"]:>+12.2f}%')
    report_lines.append('-' * 90)

    wheat_classes = sorted([c for c in head_classes if c.startswith('Wheat_')])
    for wc in wheat_classes:
        cid = head_classes[wc]
        new_acc = per_class_acc.get(cid, 0.0)
        old_acc = old_metrics.get(wc, 0.0)
        delta = new_acc - old_acc
        report_lines.append(f'{wc:<30} | {old_acc:>20.2f}% | {new_acc:>20.2f}% | {delta:>+12.2f}%')

    report_lines.append('-' * 90)

    report_text = '\n'.join(report_lines)
    print('\n' + report_text)

    # Save to report files
    with open(REPORT_FILE, 'w') as f:
        f.write(report_text + '\n')
    with open(os.path.join(OUTPUT_DIR, 'rebaseline_report.txt'), 'w') as f:
        f.write(report_text + '\n')

    print(f'\n✓ Saved re-baseline report to {REPORT_FILE}')

if __name__ == '__main__':
    main()
