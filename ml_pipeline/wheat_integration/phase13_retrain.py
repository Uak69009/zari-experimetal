import os, json, sys, time, re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, roc_auc_score

OUTPUT_DIR = 'ml_pipeline/wheat_integration'
MODELS_DIR = 'ml_pipeline/models'
LOGS_DIR = 'ml_pipeline/logs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

V2_CSV = 'ml_pipeline/data/dataset_final_training_v2.csv'
CLASS_MAP_FILE = 'ml_pipeline/data/class_map_final.json'
CLASS_WEIGHTS_V2_FILE = 'ml_pipeline/data/class_weights_v2.json'

PHASE1_V2_MODEL_PATH = os.path.join(MODELS_DIR, 'phase1_backbone_v2.pth')
PHASE2_V2_MODEL_PATH = os.path.join(MODELS_DIR, 'phase2_edl_model_v2.pth')

PHASE1_HISTORY_JSON = os.path.join(LOGS_DIR, 'phase1_v2_history.json')
PHASE2_HISTORY_JSON = os.path.join(LOGS_DIR, 'phase2_v2_history.json')
COMPARISON_REPORT_TXT = 'ml_pipeline/retrain_comparison.txt'

NUM_HEAD_CLASSES = 67
TOTAL_CLASSES = 106

class GenericImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df['image_path'].astype(str).tolist()
        self.labels = self.df[label_col].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_str = self.image_paths[idx]
        lbl = self.labels[idx]
        img = Image.open(path_str).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, lbl

def edl_kl_divergence(alpha: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
    device = alpha.device
    num_classes = alpha.size(1)
    alpha_tilde = target_one_hot + (1.0 - target_one_hot) * alpha
    sum_alpha_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    kl = (
        torch.lgamma(sum_alpha_tilde)
        - torch.lgamma(torch.tensor(float(num_classes), device=device))
        - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        + torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)),
            dim=1,
            keepdim=True,
        )
    )
    return kl.squeeze(-1)

def edl_loss_fn(logits: torch.Tensor, target_labels: torch.Tensor, epoch: int, max_epochs: int, weights: torch.Tensor | None = None):
    num_classes = logits.size(1)
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)

    uncertainty = float(num_classes) / S.squeeze(-1)
    mean_uncertainty = torch.mean(uncertainty)

    target_one_hot = F.one_hot(target_labels, num_classes=num_classes).float()
    ace_loss = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    annealing_coef = min(1.0, float(epoch + 1) / float(max_epochs))
    kl_loss = edl_kl_divergence(alpha, target_one_hot)
    loss_per_sample = ace_loss + annealing_coef * kl_loss

    if weights is not None:
        sample_weights = weights[target_labels]
        loss_per_sample = loss_per_sample * sample_weights

    return torch.mean(loss_per_sample), mean_uncertainty

def run_phase1_retrain(device, df_v2, class_map):
    print('\n=====================================================================')
    print('STAGE 1: PHASE 1 BACKBONE RETRAINING (106 CLASSES, 10 EPOCHS)')
    print('=====================================================================\n')

    # Load 106-class mapping
    head_classes = class_map['head_classes']
    pretrain_classes = class_map['pretrain_classes']
    
    # Assign uniform 0..105 index for all rows
    all_class_names = list(head_classes.keys()) + list(pretrain_classes.keys())
    master_106_map = {name: idx for idx, name in enumerate(sorted(list(set(all_class_names))))}
    
    df_v2['master_106_id'] = df_v2['class_name'].map(master_106_map)

    train_df = df_v2[df_v2['split'] == 'train'].copy()
    val_df = df_v2[df_v2['split'] == 'val'].copy()

    print(f'✓ Phase 1 Train Samples (106 classes): {len(train_df):,}')
    print(f'✓ Phase 1 Val Samples   (106 classes): {len(val_df):,}')

    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = GenericImageDataset(train_df, 'master_106_id', transform=train_transform)
    val_ds = GenericImageDataset(val_df, 'master_106_id', transform=val_transform)

    batch_size = 64 if device.type == 'cuda' else 16
    num_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

    model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True, num_classes=TOTAL_CLASSES)
    model = model.to(device)

    epochs = 10
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    p1_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'time_seconds': []}

    best_val_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss_avg = val_loss / len(val_loader)
        val_acc = (correct / total) if total > 0 else 0.0
        dur = time.time() - t0

        p1_history['epoch'].append(epoch + 1)
        p1_history['train_loss'].append(round(train_loss, 4))
        p1_history['val_loss'].append(round(val_loss_avg, 4))
        p1_history['val_acc'].append(round(val_acc * 100.0, 2))
        p1_history['time_seconds'].append(round(dur, 2))

        tag = "⭐ BEST" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save backbone weights
            backbone_state = model.state_dict()
            torch.save(backbone_state, PHASE1_V2_MODEL_PATH)
            torch.save(backbone_state, os.path.join(OUTPUT_DIR, 'phase1_backbone_v2.pth'))

        print(f"Phase 1 Epoch {epoch+1:02d}/10 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc*100:.2f}% | Time: {dur:.1f}s {tag}")
        scheduler.step()

    with open(PHASE1_HISTORY_JSON, 'w') as f:
        json.dump(p1_history, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'phase1_v2_history.json'), 'w') as f:
        json.dump(p1_history, f, indent=2)

    print(f'✓ Phase 1 Retraining Complete. Saved backbone weights to {PHASE1_V2_MODEL_PATH}')

def run_phase2_retrain(device, df_v2, class_map, weights_data):
    print('\n=====================================================================')
    print('STAGE 2: PHASE 2 EDL HEAD RETRAINING (67 FIELD CLASSES, EARLY STOPPING)')
    print('=====================================================================\n')

    field_sources = ['plantcity', 'nwrd', 'plantdoc', 'Bangladesh', 'CGIAR']
    train_df = df_v2[(df_v2['split'] == 'train') & (df_v2['source_dataset'].isin(field_sources)) & (df_v2['class_id'] >= 0)].copy()
    val_df = df_v2[(df_v2['split'] == 'val') & (df_v2['class_id'] >= 0)].copy()

    print(f'✓ Phase 2 Field Train Samples: {len(train_df):,}')
    print(f'✓ Phase 2 Field Val Samples  : {len(val_df):,}')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = GenericImageDataset(train_df, 'class_id', transform=train_transform)
    val_ds = GenericImageDataset(val_df, 'class_id', transform=val_transform)

    batch_size = 64 if device.type == 'cuda' else 16
    num_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

    # Build model & load Phase 1 v2 backbone
    model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False)
    model.reset_classifier(0)

    backbone_state = torch.load(PHASE1_V2_MODEL_PATH, map_location='cpu')
    model.load_state_dict(backbone_state, strict=False)
    model.classifier = nn.Linear(1280, NUM_HEAD_CLASSES)
    model = model.to(device)

    # Class weights tensor
    hw = [weights_data['head_weights'][str(i)] for i in range(NUM_HEAD_CLASSES)]
    weights_tensor = torch.tensor(hw, dtype=torch.float32).to(device)

    epochs = 10
    patience = 2
    no_imp = 0
    best_val_acc = 0.0

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    p2_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'mean_uncertainty': [], 'time_seconds': []}

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(images)
                loss, _ = edl_loss_fn(logits, labels, epoch, epochs, weights=weights_tensor)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_u = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                    logits = model(images)
                    loss, u = edl_loss_fn(logits, labels, epoch, epochs, weights=None)
                val_loss += loss.item()
                val_u += u.item()
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                preds = alpha.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss_avg = val_loss / len(val_loader)
        val_u_avg = val_u / len(val_loader)
        val_acc = (correct / total) if total > 0 else 0.0
        dur = time.time() - t0

        p2_history['epoch'].append(epoch + 1)
        p2_history['train_loss'].append(round(train_loss, 4))
        p2_history['val_loss'].append(round(val_loss_avg, 4))
        p2_history['val_acc'].append(round(val_acc * 100.0, 2))
        p2_history['mean_uncertainty'].append(round(val_u_avg, 4))
        p2_history['time_seconds'].append(round(dur, 2))

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            no_imp = 0
            tag = "⭐ BEST"
            torch.save(model.state_dict(), PHASE2_V2_MODEL_PATH)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'phase2_edl_model_v2.pth'))
        else:
            no_imp += 1
            tag = f"(no imp: {no_imp}/{patience})"

        print(f"Phase 2 Epoch {epoch+1:02d}/10 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc*100:.2f}% | Uncertainty: {val_u_avg:.4f} | Time: {dur:.1f}s {tag}")
        scheduler.step()

        if no_imp >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    with open(PHASE2_HISTORY_JSON, 'w') as f:
        json.dump(p2_history, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'phase2_v2_history.json'), 'w') as f:
        json.dump(p2_history, f, indent=2)

    print(f'✓ Phase 2 Retraining Complete. Saved model checkpoint to {PHASE2_V2_MODEL_PATH}')

def evaluate_retrained_model(device, df_v2, class_map):
    print('\n=====================================================================')
    print('STAGE 3: EVALUATING RETRAINED MODEL (v2) ON NEW TEST SET (6,709 IMAGES)')
    print('=====================================================================\n')

    head_classes = class_map['head_classes']
    id_to_name = {v: k for k, v in head_classes.items()}

    test_df = df_v2[(df_v2['split'] == 'test') & (df_v2['class_id'] >= 0)].copy().reset_index(drop=True)
    test_ds = GenericImageDataset(test_df, 'class_id', transform=transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    batch_size = 64 if device.type == 'cuda' else 16
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=(device.type=='cuda'))

    model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False)
    model.reset_classifier(0)
    model.classifier = nn.Linear(1280, NUM_HEAD_CLASSES)

    model.load_state_dict(torch.load(PHASE2_V2_MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_uncertainties = []

    t0 = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(images)
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                uncertainties = float(NUM_HEAD_CLASSES) / S.squeeze(-1)
                preds = probs.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_uncertainties.extend(uncertainties.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_uncertainties = np.array(all_uncertainties)

    v2_acc = accuracy_score(all_targets, all_preds) * 100.0
    v2_u = np.mean(all_uncertainties)
    try:
        v2_auroc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except Exception:
        v2_auroc = np.nan

    # Phase 12 Baseline metrics (evaluated on same test set)
    p12_baseline_acc = 98.43
    p12_baseline_auroc = 0.9993

    report_lines = []
    report_lines.append("=====================================================================")
    report_lines.append("ZARI.ai -- PHASE 13 RETRAINED MODEL (v2) EVALUATION & COMPARISON")
    report_lines.append("=====================================================================")
    report_lines.append(f"Evaluated Model Checkpoint : {PHASE2_V2_MODEL_PATH}")
    report_lines.append(f"Evaluation Test Set        : {V2_CSV} (split == 'test')")
    report_lines.append(f"Total Test Field Images    : {len(test_df):,} images")
    report_lines.append(f"Retrained Overall Accuracy : {v2_acc:.2f}% (vs Phase 12 Baseline: {p12_baseline_acc:.2f}%)")
    report_lines.append(f"Retrained Overall AUROC    : {v2_auroc:.4f} (vs Phase 12 Baseline: {p12_baseline_auroc:.4f})")
    report_lines.append(f"Mean Uncertainty           : {v2_u:.4f}\n")

    report_lines.append("=====================================================================")
    report_lines.append("FULL 67-CLASS COMPARISON TABLE (Phase 12 Re-baseline vs. Retrained v2)")
    report_lines.append("=====================================================================")
    report_lines.append(f'{"Class Name":<30} | {"Phase 12 Baseline":<22} | {"Retrained v2 Model":<22} | {"Net Delta"}')
    report_lines.append('-' * 90)

    report_lines.append(f'{"Overall Test Accuracy":<30} | {p12_baseline_acc:>20.2f}% | {v2_acc:>20.2f}% | {v2_acc - p12_baseline_acc:>+12.2f}%')
    report_lines.append('-' * 90)

    for cid in range(NUM_HEAD_CLASSES):
        cname = id_to_name[cid]
        mask = (all_targets == cid)
        class_acc = (all_preds[mask] == cid).mean() * 100.0 if mask.sum() > 0 else 0.0
        report_lines.append(f'{cname:<30} | {"N/A":>22} | {class_acc:>20.2f}% | {"-":>12}')

    report_text = '\n'.join(report_lines)
    print('\n' + report_text)

    with open(COMPARISON_REPORT_TXT, 'w') as f:
        f.write(report_text + '\n')
    with open(os.path.join(OUTPUT_DIR, 'retrain_comparison.txt'), 'w') as f:
        f.write(report_text + '\n')

    print(f'\n✓ Saved retrain comparison report to {COMPARISON_REPORT_TXT}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using compute device: {device}')
    if device.type == 'cuda':
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

    df_v2 = pd.read_csv(V2_CSV, low_memory=False)
    with open(CLASS_MAP_FILE) as f:
        class_map = json.load(f)
    with open(CLASS_WEIGHTS_V2_FILE) as f:
        weights_data = json.load(f)

    # Step 1: Retrain Phase 1 Backbone
    run_phase1_retrain(device, df_v2, class_map)

    # Step 2: Retrain Phase 2 EDL Head
    run_phase2_retrain(device, df_v2, class_map, weights_data)

    # Step 3: Evaluate Retrained Model on New Test Set
    evaluate_retrained_model(device, df_v2, class_map)

if __name__ == '__main__':
    main()
