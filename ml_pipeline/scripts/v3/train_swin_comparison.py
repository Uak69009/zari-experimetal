"""
ZARI.ai — Swin-Tiny Disease Classifier Training
Architecture comparison vs frozen EfficientNetV2-B2 Model B.

RULES:
  - Model B is NOT touched. Read-only reference only.
  - Same EDL loss, same augmentation, same optimizer/scheduler/early-stopping
    as confirmed in Phase 1 baseline audit.
  - Separate SCRC calibration on Swin val uncertainties.
  - All outputs -> ml_pipeline/models/swin_comparison/
"""

import os, sys, json, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.models import swin_t, Swin_T_Weights
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, balanced_accuracy_score, roc_auc_score
)

REPO_ROOT    = Path("/home/hammad/Desktop/project zari - experimental")
DATA_DIR     = REPO_ROOT / "ml_pipeline" / "data"
V4_CSV_PATH  = DATA_DIR / "dataset_3crop_final_v4_split.csv"
EFFNET_PATH  = DATA_DIR / "reports_v3" / "model_b_test_metrics.json"
OUT_DIR      = REPO_ROOT / "ml_pipeline" / "models" / "swin_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Fixed seed: {seed}")

class DiseaseDataset(Dataset):
    def __init__(self, df, label_mapping, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_mapping = label_mapping
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            img_rgb = img.convert("RGB")
        img_tensor = self.transform(img_rgb) if self.transform else T.functional.to_dtype(T.functional.to_image(img_rgb), torch.float32, scale=True)
        return img_tensor, torch.tensor(self.label_mapping[row["class_name"]], dtype=torch.long)

class EDLSwinTiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_feat = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_feat, num_classes))
    def forward(self, x):
        logits = self.backbone(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        uncertainty = logits.shape[1] / S
        return logits, evidence, alpha, S, probs, uncertainty.squeeze(-1)
    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if "backbone.head" not in name: p.requires_grad = False
    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True
    def backbone_params(self): return [p for n,p in self.named_parameters() if "backbone.head" not in n]
    def head_params(self):     return [p for n,p in self.named_parameters() if "backbone.head" in n]

class EDLLoss(nn.Module):
    def __init__(self, class_weights=None, kl_penalty=0.1):
        super().__init__()
        self.class_weights = class_weights
        self.kl_penalty = kl_penalty
    def forward(self, alpha, target, epoch=1):
        K = alpha.shape[1]
        t_oh = F.one_hot(target, K).float()
        S = alpha.sum(dim=1, keepdim=True)
        ll = (t_oh * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
        if self.class_weights is not None:
            ll = ll * self.class_weights[target]
        at = t_oh + (1-t_oh)*alpha
        St = at.sum(dim=1, keepdim=True)
        kl = (torch.lgamma(St) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
              - torch.lgamma(at).sum(dim=1, keepdim=True)
              + ((at-1)*(torch.digamma(at)-torch.digamma(St))).sum(dim=1, keepdim=True))
        anneal = min(1.0, epoch/10.0)*self.kl_penalty
        return torch.mean(ll + anneal*kl.squeeze(-1))

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience=patience; self.min_delta=min_delta
        self.counter=0; self.best=None; self.early_stop=False
    def __call__(self, s):
        if self.best is None: self.best=s; return True
        if s-self.best > self.min_delta: self.best=s; self.counter=0; return True
        self.counter+=1
        if self.counter>=self.patience: self.early_stop=True
        return False

def diagnose(tl,vl,ta,va,tf,vf,hist):
    if math.isnan(tl) or math.isinf(tl): return "UNSTABLE"
    if ta<.5 and va<.5: return "UNDERFITTING_WARNING"
    if len(hist)>=2 and vl>hist[-1]["val_loss"] and tl<hist[-1]["train_loss"] and ta-va>0.12: return "OVERFITTING_WARNING"
    return "HEALTHY"

def measure_latency(model, n=100, sz=256):
    model.eval()
    d = torch.randn(1,3,sz,sz).cuda()
    with torch.no_grad():
        for _ in range(10): model(d)
    torch.cuda.synchronize(); t0=time.perf_counter()
    with torch.no_grad():
        for _ in range(n): model(d)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1000

def calibrate_scrc(uncs, corrects, crop, target_risk=0.02):
    u=np.array(uncs); c=np.array(corrects); N=len(u)
    thresholds=np.linspace(0.005,u.max(),300)
    table=[]
    for t in thresholds:
        m=u<=t; n_acc=m.sum()
        if n_acc==0: continue
        cov=n_acc/N; sacc=c[m].mean(); risk=1-sacc
        table.append({"threshold":round(float(t),5),"coverage":round(float(cov),4),
                      "selective_accuracy":round(float(sacc),4),"selective_risk":round(float(risk),4),"n_accepted":int(n_acc)})
    df=pd.DataFrame(table)
    safe=df[df["selective_risk"]<=target_risk]
    best=safe.loc[safe["coverage"].idxmax()] if len(safe) else df.loc[df["selective_risk"].idxmin()]
    print(f"\n  [{crop}] SCRC calibration:")
    print(f"    Threshold={best['threshold']:.5f}  Coverage={best['coverage']*100:.2f}%  SelRisk={best['selective_risk']*100:.2f}%  Accepted={best['n_accepted']}/{N}")
    return {"edl_uncertainty_threshold":float(best["threshold"]),"coverage":float(best["coverage"]),
            "selective_accuracy":float(best["selective_accuracy"]),"selective_risk":float(best["selective_risk"]),
            "n_accepted":int(best["n_accepted"]),"n_total":N}, df.to_dict("records")

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256,256),antialias=True),
    T.RandomHorizontalFlip(p=0.50),
    T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
    T.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.90,1.10)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    T.GaussianBlur(kernel_size=(3,3), sigma=(0.1,1.0)),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    T.RandomErasing(p=0.10, scale=(0.02,0.10), value=0),
])
VAL_TRANSFORM = T.Compose([
    T.Resize((256,256),antialias=True),
    T.ToImage(), T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def train_crop(crop, K, mapping, weights, tr_df, va_df, te_df, max_epochs=20):
    print(f"\n{'='*70}")
    print(f"  SWIN-TINY | {crop.upper()} | {K} classes")
    print(f"{'='*70}")
    tr = tr_df[(tr_df["crop"]==crop)&(tr_df["class_name"].isin(mapping))].copy()
    va = va_df[(va_df["crop"]==crop)&(va_df["class_name"].isin(mapping))].copy()
    te = te_df[(te_df["crop"]==crop)&(te_df["class_name"].isin(mapping))].copy()
    print(f"  Train:{len(tr):,}  Val:{len(va):,}  Test:{len(te):,}")

    w_t = torch.tensor([weights[i] for i in range(K)], dtype=torch.float32).cuda()
    tr_dl = DataLoader(DiseaseDataset(tr,mapping,TRAIN_TRANSFORM), batch_size=64, shuffle=True,  num_workers=8, pin_memory=True, persistent_workers=True)
    va_dl = DataLoader(DiseaseDataset(va,mapping,VAL_TRANSFORM),   batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    te_dl = DataLoader(DiseaseDataset(te,mapping,VAL_TRANSFORM),   batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model = EDLSwinTiny(K).cuda()
    crit  = EDLLoss(class_weights=w_t, kl_penalty=0.1)
    scaler = torch.amp.GradScaler("cuda")
    es    = EarlyStopping(patience=5, min_delta=0.001)
    best_ckpt = OUT_DIR / f"swin_{crop.lower()}_disease.pth"
    hist, best_f1, best_ep = [], 0.0, 0
    STAGE1 = 3

    if best_ckpt.exists():
        print(f"  ✓ Existing checkpoint found: {best_ckpt.name}. Skipping training.")
        sd = torch.load(best_ckpt, map_location="cuda", weights_only=False)
        best_ep = sd.get("epoch", 0)
        best_f1 = sd.get("best_val_macro_f1", 0.0)
        hist_file = OUT_DIR / f"swin_{crop.lower()}_training_history.csv"
        if hist_file.exists():
            hist = pd.read_csv(hist_file).to_dict("records")
        else:
            hist = []
    else:
        for ep in range(1, max_epochs+1):
            t0 = time.time()
            stage = "STAGE_1_HEAD_WARMUP" if ep<=STAGE1 else "STAGE_2_FINE_TUNING"
            if ep==1:
                model.freeze_backbone()
                opt = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
                sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2, min_lr=1e-7)
            elif ep==STAGE1+1:
                model.unfreeze_all()
                opt = optim.AdamW([{"params":model.backbone_params(),"lr":1e-4},
                                   {"params":model.head_params(),"lr":1e-3}], weight_decay=1e-4)
                sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=2, min_lr=1e-7)

            model.train(); tl,tp,tt = 0.0,[],[]
            for x,y in tr_dl:
                x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    _,_,alpha,_,probs,_ = model(x)
                    loss = crit(alpha,y,epoch=ep)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
                tl+=loss.item(); tp.extend(torch.argmax(probs,1).cpu().numpy()); tt.extend(y.cpu().numpy())
            tl/=len(tr_dl); ta=accuracy_score(tt,tp); tf=f1_score(tt,tp,average="macro",zero_division=0)

            model.eval(); vl,vp,vt,vu = 0.0,[],[],[]
            with torch.no_grad():
                for x,y in va_dl:
                    x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                    with torch.amp.autocast("cuda"):
                        _,_,alpha,_,probs,unc = model(x)
                        loss=crit(alpha,y,epoch=ep)
                    vl+=loss.item(); vp.extend(torch.argmax(probs,1).cpu().numpy()); vt.extend(y.cpu().numpy()); vu.extend(unc.cpu().numpy())
            vl/=len(va_dl); va_acc=accuracy_score(vt,vp); vf=f1_score(vt,vp,average="macro",zero_division=0)
            sch.step(vl)
            lr_bb = opt.param_groups[0]["lr"] if len(opt.param_groups)>1 else 0.0
            lr_hd = opt.param_groups[-1]["lr"]
            diag = diagnose(tl,vl,ta,va_acc,tf,vf,hist)
            hist.append({"epoch":ep,"stage":stage,"train_loss":round(tl,4),"val_loss":round(vl,4),
                         "train_accuracy":round(ta,4),"val_accuracy":round(va_acc,4),
                         "train_macro_f1":round(tf,4),"val_macro_f1":round(vf,4),
                         "mean_val_uncertainty":round(float(np.mean(vu)),4),
                         "learning_rate_backbone":lr_bb,"learning_rate_head":lr_hd,
                         "generalization_gap":round(tf-vf,4),"diagnostic_state":diag})
            print(f"[{crop}] Ep{ep:02d}/{max_epochs} [{stage[:7]}] | Tr:{tl:.4f}/{ta*100:.1f}%/F1={tf:.4f} | Va:{vl:.4f}/{va_acc*100:.1f}%/F1={vf:.4f} | {diag} ({time.time()-t0:.1f}s)")

            ckpt = {"epoch":ep,"stage":stage,"model_state_dict":model.state_dict(),
                    "optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":sch.state_dict(),
                    "val_macro_f1":vf,"val_accuracy":va_acc,"class_mapping":mapping,"crop":crop,
                    "architecture":"Swin-Tiny","config":{"batch_size":64,"seed":42}}
            if es(vf):
                best_f1=vf; best_ep=ep
                ckpt["best_val_macro_f1"]=best_f1
                torch.save(ckpt, best_ckpt)
                print(f"  ★ NEW BEST [{crop}] Ep{ep}  Val F1={best_f1:.4f}")
            if es.early_stop:
                hist[-1]["diagnostic_state"]="EARLY_STOPPING"
                print(f"\n✋ Early stop [{crop}] at Ep{ep}")
                break

        pd.DataFrame(hist).to_csv(OUT_DIR/f"swin_{crop.lower()}_training_history.csv",index=False)

    # Reload best
    print(f"\nReloading best [{crop}] checkpoint (Ep{best_ep})...")
    sd = torch.load(best_ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()

    lat = measure_latency(model)
    print(f"  GPU Latency: {lat:.2f} ms/image (256x256, single batch)")

    def eval_split(loader, name):
        yp,yt,yu,yprob=[],[],[],[]
        with torch.no_grad():
            for x,y in loader:
                x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                with torch.amp.autocast("cuda"):
                    _,_,alpha,_,probs,unc=model(x)
                yp.extend(torch.argmax(probs,1).cpu().numpy())
                yt.extend(y.cpu().numpy())
                yu.extend(unc.cpu().numpy())
                yprob.append(probs.cpu().numpy())
        yp=np.array(yp); yt=np.array(yt); yu=np.array(yu); yprob=np.vstack(yprob)
        acc=float(accuracy_score(yt,yp))
        bacc=float(balanced_accuracy_score(yt,yp))
        mf1=float(f1_score(yt,yp,average="macro",zero_division=0))
        pr,re,f1,sup=precision_recall_fscore_support(yt,yp,average=None,labels=list(range(K)),zero_division=0)
        cm=confusion_matrix(yt,yp).tolist()
        try: auroc=float(roc_auc_score(yt,yprob,multi_class="ovr",average="macro"))
        except: auroc=None
        cnames=[k for k,v in sorted(mapping.items(),key=lambda x:x[1])]
        pc={cnames[i]:{"support":int(sup[i]),"precision":round(float(pr[i]),4),
                       "recall":round(float(re[i]),4),"f1":round(float(f1[i]),4)} for i in range(K)}
        unc_c=float(np.mean(yu[yp==yt])) if (yp==yt).any() else 0.0
        unc_w=float(np.mean(yu[yp!=yt])) if (yp!=yt).any() else 0.0
        return {"acc":acc,"bal_acc":bacc,"macro_f1":mf1,"auroc":auroc,
                "unc_correct":unc_c,"unc_incorrect":unc_w,"per_class":pc,"cm":cm,"y_unc":yu,"y_correct":(yp==yt).astype(int)}

    va_res = eval_split(va_dl, "val")
    te_res = eval_split(te_dl, "test")

    scrc, scrc_table = calibrate_scrc(va_res["y_unc"], va_res["y_correct"], crop)

    auroc_str = f"{te_res['auroc']:.4f}" if te_res['auroc'] is not None else "N/A"
    print(f"\n{'='*70}")
    print(f"  SWIN [{crop}] TEST RESULTS (Epoch {best_ep})")
    print(f"{'='*70}")
    print(f"  Acc={te_res['acc']*100:.2f}%  BalAcc={te_res['bal_acc']*100:.2f}%  MacroF1={te_res['macro_f1']:.4f}  AUROC={auroc_str}")
    print(f"  Latency={lat:.2f}ms  SCRC_thr={scrc['edl_uncertainty_threshold']:.5f}  Cov={scrc['coverage']*100:.2f}%  SelRisk={scrc['selective_risk']*100:.2f}%")
    print("  Per-class test F1:")
    for cn,m in te_res["per_class"].items():
        print(f"    {cn:<45}  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  n={m['support']}")
    print("  Confusion matrix:")
    for row in te_res["cm"]: print(f"    {row}")

    va_clean = {k:v for k,v in va_res.items() if k not in ("y_unc","y_correct")}
    te_clean = {k:v for k,v in te_res.items() if k not in ("y_unc","y_correct")}
    return {"crop":crop,"best_epoch":best_ep,"best_val_f1":best_f1,"latency_ms":lat,
            "val":va_clean,"test":te_clean,"scrc":scrc,"scrc_table":scrc_table,"history":hist}

def main():
    print("="*70)
    print("  ZARI.ai — Swin-Tiny Disease Classifiers (vs Frozen Model B EfficientNet)")
    print("="*70)
    set_seed(42)

    df = pd.read_csv(V4_CSV_PATH, low_memory=False)
    tr_df = df[df["split"]=="train"].copy()
    va_df = df[df["split"]=="val"].copy()
    te_df = df[df["split"]=="test"].copy()
    print(f"Manifest: {len(df):,}  Train:{len(tr_df):,}  Val:{len(va_df):,}  Test:{len(te_df):,}")

    tomato_map = {"Tomato_Bacterial_Spot":0,"Tomato_Early_Blight":1,"Tomato_Fusarium_Wilt":2,
                  "Tomato_Healthy":3,"Tomato_Late_Blight":4,"Tomato_Leaf_Mold":5,
                  "Tomato_Miner":6,"Tomato_Mosaic_Virus":7,"Tomato_Septoria_Leaf_Spot":8,
                  "Tomato_Spider_Mites":9,"Tomato_Target_Spot":10,"Tomato_Verticillium_Wilt":11,
                  "Tomato_Yellow_Leaf_Curl_Virus":12}
    potato_map = {"Potato_Early_Blight":0,"Potato_Late_Blight":1,"Potato_Healthy":2}
    pepper_map = {"Pepper_Bacterial_Spot":0,"Pepper_Cercospora_Leaf_Spot":1,"Pepper_Healthy":2,
                  "Pepper_Leaf_Curl":3,"Pepper_Nutrition_Deficiency":4,"Pepper_Powdery_Mildew":5}

    tomato_w = {0:0.6833,1:0.8704,2:6.6286,3:0.7586,4:0.7848,5:0.8711,6:1.4201,7:6.3176,8:0.7512,9:1.1645,10:1.9184,11:5.1911,12:0.3663}
    potato_w = {0:0.7835,1:0.8481,2:1.8365}
    pepper_w = {0:0.3326,1:0.8932,2:0.9064,3:3.2525,4:3.1063,5:7.0887}

    results = {}
    for crop,K,mapping,weights in [
        ("Tomato",13,tomato_map,tomato_w),
        ("Potato", 3,potato_map,potato_w),
        ("Pepper", 6,pepper_map,pepper_w),
    ]:
        results[crop] = train_crop(crop,K,mapping,weights,tr_df,va_df,te_df,max_epochs=20)

    # Save SCRC
    scrc_out = {c:r["scrc"] for c,r in results.items()}
    with open(OUT_DIR/"swin_scrc_thresholds.json","w") as f: json.dump(scrc_out,f,indent=2)
    print(f"\n✓ SCRC thresholds saved: ml_pipeline/models/swin_comparison/swin_scrc_thresholds.json")

    # Save test metrics
    metrics_out = {}
    for c,r in results.items():
        metrics_out[c] = {"best_epoch":r["best_epoch"],"latency_ms":r["latency_ms"],
                          "val":{k:v for k,v in r["val"].items() if k!="cm"},
                          "test":{k:v for k,v in r["test"].items() if k!="cm"},
                          "confusion_matrix_test":r["test"]["cm"]}
    with open(OUT_DIR/"swin_test_metrics.json","w") as f: json.dump(metrics_out,f,indent=2)
    print(f"✓ Test metrics saved: ml_pipeline/models/swin_comparison/swin_test_metrics.json")

    # Load EfficientNet reference
    with open(EFFNET_PATH) as f: ef_ref = json.load(f)

    # Final comparison print
    print("\n"+"="*70)
    print("  FINAL: MODEL B (EfficientNetV2-B2) vs SWIN-TINY — SIDE BY SIDE")
    print("="*70)
    for crop,res in results.items():
        ef = ef_ref[crop]
        ef_f1  = ef["test"]["macro_f1"]
        sw_f1  = res["test"]["macro_f1"]
        sw_auroc = res["test"]["auroc"]
        sw_auroc_str = f"{sw_auroc:.4f}" if sw_auroc is not None else "N/A"
        print(f"\n  ── {crop} ──────────────────────────────────────────────────────")
        print(f"  {'Metric':<35} {'EfficientNetV2-B2':>20} {'Swin-Tiny':>12} {'Delta':>10}")
        print(f"  {'-'*77}")
        print(f"  {'Test Macro F1':<35} {ef_f1:>20.4f} {sw_f1:>12.4f} {sw_f1-ef_f1:>+10.4f}")
        print(f"  {'Test Accuracy':<35} {ef['test']['acc']*100:>19.2f}% {res['test']['acc']*100:>11.2f}%")
        print(f"  {'Balanced Accuracy':<35} {ef['test']['bal_acc']*100:>19.2f}% {res['test']['bal_acc']*100:>11.2f}%")
        print(f"  {'Test Macro AUROC':<35} {'(not in EfficientNet rpt)':>20} {sw_auroc_str:>12}")
        print(f"  {'Best Epoch':<35} {ef['best_epoch']:>20} {res['best_epoch']:>12}")
        print(f"  {'GPU Latency (ms/img)':<35} {'—':>20} {res['latency_ms']:>11.2f}ms")
        print(f"  {'SCRC Threshold':<35} {'0.4500 (calibrated)':>20} {res['scrc']['edl_uncertainty_threshold']:>12.5f}")
        print(f"  {'SCRC Coverage (val)':<35} {'97.40%':>20} {res['scrc']['coverage']*100:>11.2f}%")
        print(f"  {'SCRC Sel. Risk (val)':<35} {'1.20%':>20} {res['scrc']['selective_risk']*100:>11.2f}%")
        print(f"\n  Per-class F1 (TEST):")
        for cname,sw_m in res["test"]["per_class"].items():
            ef_c = ef["per_class"].get(cname,{}).get("f1",float("nan"))
            print(f"    {cname:<45} Eff={ef_c:.4f}  Swin={sw_m['f1']:.4f}  d={sw_m['f1']-ef_c:+.4f}  n={sw_m['support']}")
        print(f"\n  Confusion matrix (Swin, test):")
        for row in res["test"]["cm"]: print(f"    {row}")

    print("\n"+"-"*70)
    print("  SUMMARY — Macro F1 (Test)")
    print(f"  {'Crop':<12} {'EfficientNet':>15} {'Swin-Tiny':>12} {'Delta':>10} {'Winner':>14}")
    print("  "+"-"*63)
    for crop,res in results.items():
        ef_f1=ef_ref[crop]["test"]["macro_f1"]; sw_f1=res["test"]["macro_f1"]; d=sw_f1-ef_f1
        w="EfficientNet" if d<-0.001 else ("Swin-Tiny" if d>0.001 else "Tie")
        print(f"  {crop:<12} {ef_f1:>15.4f} {sw_f1:>12.4f} {d:>+10.4f} {w:>14}")

    print(f"\n  All files in: ml_pipeline/models/swin_comparison/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name} ({f.stat().st_size//1024} KB)")
    print("\nSTOP — Phase complete. Awaiting instruction.")

if __name__=="__main__":
    main()
