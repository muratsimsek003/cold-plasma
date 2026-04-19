"""
Vision Transformer (ViT) for Cold Plasma Treatment Eligibility Classification
Binary classification: Plasma-Treatable (malignant/pre-malignant) vs Benign

Clinical context: Cold atmospheric plasma selectively targets malignant cells.
Accurate pre-treatment classification determines plasma treatment eligibility.

Dataset: HAM10000 (dermoscopic images)
  - Plasma-treatable: melanoma (MEL), basal cell carcinoma (BCC), actinic keratoses (AK)
  - Benign (no treatment): melanocytic nevi (NV), benign keratosis (BKL), dermatofibroma (DF), vascular (VASC)

Models: ViT-Tiny (scratch), ViT-Tiny (pretrained), ResNet-18 (scratch)
Evaluation: 5-Fold Stratified Cross Validation
"""

import os, sys, json, time, random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, accuracy_score)
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import timm

# ============================================================
# Config
# ============================================================
# Mapping: original class -> binary label
# 1 = Plasma-treatable (malignant/pre-malignant)
# 0 = Benign (no plasma treatment)
ORIGINAL_CLASSES = {
    "actinic_keratoses": 1,         # Pre-malignant -> treatable
    "basal_cell_carcinoma": 1,      # Malignant -> treatable
    "melanoma": 1,                  # Malignant -> treatable
    "benign_keratosis-like_lesions": 0,  # Benign
    "dermatofibroma": 0,            # Benign
    "melanocytic_Nevi": 0,          # Benign
    "vascular_lesions": 0,          # Benign
}
BINARY_NAMES = ["Benign", "Plasma-Treatable"]
NUM_CLASSES = 2
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
K_FOLDS = 3
DATA_DIR = "skin_dataset/images"
OUT_DIR = "results_plasma_skin"
MAX_PER_CLASS = 200  # per original class
DEVICE = torch.device("cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ============================================================
# Dataset
# ============================================================
class SkinDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

def load_data(data_dir, max_per_class=200):
    paths, labels, orig_classes = [], [], []
    for cls_name, binary_label in ORIGINAL_CLASSES.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir): continue
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith((".jpg",".png",".bmp"))])
        random.seed(SEED)
        random.shuffle(files)
        selected = files[:max_per_class]
        for f in selected:
            paths.append(os.path.join(cls_dir, f))
            labels.append(binary_label)
            orig_classes.append(cls_name)
    return paths, labels, orig_classes

def get_weighted_sampler(labels):
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ============================================================
# Model Builders
# ============================================================
def build_vit_tiny_scratch():
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)

def build_vit_tiny_pretrained():
    return timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=NUM_CLASSES)

def build_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

MODEL_BUILDERS = {
    "ViT-Tiny (scratch)": build_vit_tiny_scratch,
    "ViT-Tiny (pretrained)": build_vit_tiny_pretrained,
    "ResNet-18 (scratch)": build_resnet18,
}

# ============================================================
# Training & Evaluation
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labs)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()*imgs.size(0)
        correct += out.argmax(1).eq(labs).sum().item()
        total += labs.size(0)
    return loss_sum/total, 100.*correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            loss = criterion(out, labs)
            loss_sum += loss.item()*imgs.size(0)
            pred = out.argmax(1)
            prob = torch.softmax(out, dim=1)[:, 1]  # prob of class 1 (treatable)
            correct += pred.eq(labs).sum().item()
            total += labs.size(0)
            preds.extend(pred.cpu().numpy())
            trues.extend(labs.cpu().numpy())
            probs.extend(prob.cpu().numpy())
    return loss_sum/total, 100.*correct/total, np.array(preds), np.array(trues), np.array(probs)

# ============================================================
# K-Fold Cross Validation
# ============================================================
def run_kfold(model_name, builder_fn, all_paths, all_labels, device):
    print(f"\n{'='*70}", flush=True)
    print(f"  {model_name} - {K_FOLDS}-Fold Cross Validation", flush=True)
    print(f"{'='*70}", flush=True)

    tmp = builder_fn()
    params = sum(p.numel() for p in tmp.parameters())
    print(f"  Parameters: {params:,}", flush=True)
    del tmp

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    all_fold_preds, all_fold_trues, all_fold_probs = [], [], []
    total_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_paths, all_labels)):
        print(f"\n  --- Fold {fold+1}/{K_FOLDS} (Train: {len(train_idx)}, Test: {len(test_idx)}) ---", flush=True)

        np.random.shuffle(train_idx)
        n_val = int(len(train_idx) * 0.1)
        val_idx, tr_idx = train_idx[:n_val], train_idx[n_val:]

        tr_labels = [all_labels[i] for i in tr_idx]
        tr_ds = SkinDataset([all_paths[i] for i in tr_idx], tr_labels, train_tf)
        val_ds = SkinDataset([all_paths[i] for i in val_idx], [all_labels[i] for i in val_idx], test_tf)
        te_ds = SkinDataset([all_paths[i] for i in test_idx], [all_labels[i] for i in test_idx], test_tf)

        sampler = get_weighted_sampler(tr_labels)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        set_seed(SEED + fold)
        model = builder_fn().to(device)

        class_counts = Counter(tr_labels)
        total_samples = len(tr_labels)
        class_weights = torch.tensor([total_samples / (NUM_CLASSES * class_counts.get(i, 1))
                                      for i in range(NUM_CLASSES)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_acc, best_state = 0, None
        fold_hist = {"train_acc":[], "val_acc":[], "train_loss":[], "val_loss":[]}
        t0 = time.time()

        for ep in range(EPOCHS):
            tl, ta = train_epoch(model, tr_dl, criterion, optimizer, device)
            vl, va, _, _, _ = evaluate(model, val_dl, criterion, device)
            scheduler.step()
            fold_hist["train_acc"].append(ta); fold_hist["val_acc"].append(va)
            fold_hist["train_loss"].append(tl); fold_hist["val_loss"].append(vl)
            if va > best_val_acc:
                best_val_acc = va
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            if True:  # log every epoch
                print(f"    Ep {ep+1:2d}/{EPOCHS} | Train: {ta:.1f}% ({tl:.4f}) | Val: {va:.1f}% ({vl:.4f})", flush=True)

        fold_time = time.time() - t0
        model.load_state_dict(best_state); model.to(device)
        te_loss, te_acc, te_preds, te_trues, te_probs = evaluate(model, te_dl, criterion, device)

        prec, rec, f1, _ = precision_recall_fscore_support(te_trues, te_preds, average="binary")
        try:
            auc = roc_auc_score(te_trues, te_probs)
        except:
            auc = 0.0

        # Sensitivity (recall for class 1) and Specificity (recall for class 0)
        cm_fold = confusion_matrix(te_trues, te_preds)
        tn, fp, fn, tp = cm_fold.ravel() if cm_fold.shape == (2,2) else (0,0,0,0)
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)

        print(f"    Fold {fold+1} Test: Acc={te_acc:.2f}%, F1={f1:.4f}, AUC={auc:.4f}, "
              f"Sens={sensitivity:.4f}, Spec={specificity:.4f}, Time={fold_time:.0f}s", flush=True)

        fold_results.append({
            "fold": fold+1, "test_acc": te_acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc,
            "sensitivity": sensitivity, "specificity": specificity,
            "time": fold_time, "history": fold_hist, "best_val_acc": best_val_acc,
        })
        all_fold_preds.extend(te_preds.tolist())
        all_fold_trues.extend(te_trues.tolist())
        all_fold_probs.extend(te_probs.tolist())
        del model, best_state

    total_time = time.time() - total_start

    accs = [r["test_acc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    sens = [r["sensitivity"] for r in fold_results]
    specs = [r["specificity"] for r in fold_results]
    precs = [r["precision"] for r in fold_results]

    overall_report = classification_report(all_fold_trues, all_fold_preds,
                                           target_names=BINARY_NAMES, digits=4)
    overall_cm = confusion_matrix(all_fold_trues, all_fold_preds)

    print(f"\n  {model_name} - Aggregated Results:", flush=True)
    print(f"  Accuracy:    {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%", flush=True)
    print(f"  F1-Score:    {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}", flush=True)
    print(f"  AUC:         {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}", flush=True)
    print(f"  Sensitivity: {np.mean(sens):.4f} +/- {np.std(sens):.4f}", flush=True)
    print(f"  Specificity: {np.mean(specs):.4f} +/- {np.std(specs):.4f}", flush=True)
    print(f"  Total time:  {total_time:.0f}s", flush=True)
    print(f"\n{overall_report}", flush=True)

    return {
        "model": model_name, "params": params,
        "mean_acc": np.mean(accs), "std_acc": np.std(accs),
        "mean_f1": np.mean(f1s), "std_f1": np.std(f1s),
        "mean_auc": np.mean(aucs), "std_auc": np.std(aucs),
        "mean_sensitivity": np.mean(sens), "std_sensitivity": np.std(sens),
        "mean_specificity": np.mean(specs), "std_specificity": np.std(specs),
        "mean_precision": np.mean(precs), "std_precision": np.std(precs),
        "fold_results": fold_results,
        "overall_cm": overall_cm.tolist(),
        "overall_report": overall_report,
        "total_time": total_time,
        "all_probs": all_fold_probs,
        "all_trues": all_fold_trues,
    }

# ============================================================
# Plotting
# ============================================================
def make_plots(all_results, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 1. Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for res in all_results:
        name = res["model"]
        n_ep = len(res["fold_results"][0]["history"]["train_acc"])
        avg_ta = np.mean([[r["history"]["train_acc"][e] for r in res["fold_results"]] for e in range(n_ep)], axis=1)
        avg_va = np.mean([[r["history"]["val_acc"][e] for r in res["fold_results"]] for e in range(n_ep)], axis=1)
        avg_tl = np.mean([[r["history"]["train_loss"][e] for r in res["fold_results"]] for e in range(n_ep)], axis=1)
        avg_vl = np.mean([[r["history"]["val_loss"][e] for r in res["fold_results"]] for e in range(n_ep)], axis=1)
        eps = range(1, n_ep+1)
        ax1.plot(eps, avg_ta, "-o", label=f"{name} (train)", markersize=3)
        ax1.plot(eps, avg_va, "--s", label=f"{name} (val)", markersize=3)
        ax2.plot(eps, avg_tl, "-o", label=f"{name} (train)", markersize=3)
        ax2.plot(eps, avg_vl, "--s", label=f"{name} (val)", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Training & Validation Accuracy ({K_FOLDS}-Fold CV)")
    ax1.legend(fontsize=6); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title(f"Training & Validation Loss ({K_FOLDS}-Fold CV)")
    ax2.legend(fontsize=6); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/training_curves.png", dpi=150); plt.close()

    # 2. Confusion matrices (normalized)
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 4))
    if len(all_results)==1: axes=[axes]
    for ax, r in zip(axes, all_results):
        cm = np.array(r["overall_cm"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=100)
        ax.set_title(f"{r['model']}\nAcc:{r['mean_acc']:.1f}% AUC:{r['mean_auc']:.3f}", fontsize=9)
        ax.set_xticks(range(2)); ax.set_xticklabels(BINARY_NAMES, fontsize=8)
        ax.set_yticks(range(2)); ax.set_yticklabels(BINARY_NAMES, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1f}%)", ha="center", va="center",
                        fontsize=9, color="white" if cm_norm[i,j]>50 else "black")
    plt.tight_layout(); plt.savefig(f"{out_dir}/confusion_matrices.png", dpi=150); plt.close()

    # 3. Model comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["model"] for r in all_results]
    metrics = {
        "Accuracy": ([r["mean_acc"] for r in all_results], [r["std_acc"] for r in all_results]),
        "Sensitivity": ([r["mean_sensitivity"]*100 for r in all_results], [r["std_sensitivity"]*100 for r in all_results]),
        "Specificity": ([r["mean_specificity"]*100 for r in all_results], [r["std_specificity"]*100 for r in all_results]),
        "AUC": ([r["mean_auc"]*100 for r in all_results], [r["std_auc"]*100 for r in all_results]),
    }
    x = np.arange(len(names))
    width = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, (metric_name, (vals, stds)) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, yerr=stds, capsize=3,
                      label=metric_name, color=colors[i], alpha=0.85)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                    f"{b.get_height():.1f}", ha="center", fontsize=7)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Cold Plasma Treatment Eligibility Classification ({K_FOLDS}-Fold CV)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=8); ax.set_ylim(0, 110); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{out_dir}/model_comparison.png", dpi=150); plt.close()

    # 4. ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_roc = ["#4C72B0", "#DD8452", "#55A868"]
    for r, color in zip(all_results, colors_roc):
        trues = np.array(r["all_trues"])
        probs = np.array(r["all_probs"])
        fpr, tpr, _ = roc_curve(trues, probs)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{r['model']} (AUC={r['mean_auc']:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Plasma Treatment Eligibility Classification")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/roc_curves.png", dpi=150); plt.close()

    # 5. Fold-wise accuracy boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    fold_accs = [[r["test_acc"] for r in res["fold_results"]] for res in all_results]
    bp = ax.boxplot(fold_accs, labels=[r["model"] for r in all_results], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_roc):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("Accuracy (%)"); ax.set_title(f"Accuracy Distribution Across {K_FOLDS} Folds")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{out_dir}/accuracy_boxplot.png", dpi=150); plt.close()

    print(f"  All plots saved to {out_dir}/", flush=True)

# ============================================================
# Save results table as CSV
# ============================================================
def save_results_table(all_results, out_dir):
    lines = []
    # Main comparison table
    header = "Model,Parameters,Accuracy(%),Acc_Std,F1-Score,F1_Std,AUC,AUC_Std,Sensitivity,Sens_Std,Specificity,Spec_Std,Precision,Prec_Std,Time(s)"
    lines.append(header)
    for r in all_results:
        lines.append(f"{r['model']},{r['params']},"
                     f"{r['mean_acc']:.2f},{r['std_acc']:.2f},"
                     f"{r['mean_f1']:.4f},{r['std_f1']:.4f},"
                     f"{r['mean_auc']:.4f},{r['std_auc']:.4f},"
                     f"{r['mean_sensitivity']:.4f},{r['std_sensitivity']:.4f},"
                     f"{r['mean_specificity']:.4f},{r['std_specificity']:.4f},"
                     f"{r['mean_precision']:.4f},{r['std_precision']:.4f},"
                     f"{r['total_time']:.1f}")
    with open(f"{out_dir}/results_table.csv", "w") as f:
        f.write("\n".join(lines))

    # Per-fold table
    fold_lines = ["Model,Fold,Accuracy(%),F1-Score,AUC,Sensitivity,Specificity,Time(s)"]
    for r in all_results:
        for fr in r["fold_results"]:
            fold_lines.append(f"{r['model']},{fr['fold']},"
                              f"{fr['test_acc']:.2f},{fr['f1']:.4f},"
                              f"{fr['auc']:.4f},{fr['sensitivity']:.4f},"
                              f"{fr['specificity']:.4f},{fr['time']:.1f}")
    with open(f"{out_dir}/results_per_fold.csv", "w") as f:
        f.write("\n".join(fold_lines))

    print(f"  Tables saved to {out_dir}/results_table.csv and results_per_fold.csv", flush=True)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    set_seed(SEED)
    print(f"{'='*70}", flush=True)
    print(f"  Cold Plasma Treatment Eligibility - Skin Lesion Classification", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Device: {DEVICE} | PyTorch: {torch.__version__}", flush=True)
    print(f"Config: epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}, folds={K_FOLDS}", flush=True)
    print(f"Task: Binary (Benign vs Plasma-Treatable)", flush=True)
    print(f"Max per original class: {MAX_PER_CLASS}", flush=True)

    paths, labels, orig_classes = load_data(DATA_DIR, MAX_PER_CLASS)
    print(f"\nDataset: {len(paths)} images", flush=True)
    print(f"  Original class distribution:", flush=True)
    orig_counts = Counter(orig_classes)
    for cls, cnt in sorted(orig_counts.items()):
        label = ORIGINAL_CLASSES[cls]
        print(f"    {cls}: {cnt} -> {'Treatable' if label==1 else 'Benign'}", flush=True)
    binary_counts = Counter(labels)
    print(f"  Binary distribution:", flush=True)
    print(f"    Benign: {binary_counts[0]}", flush=True)
    print(f"    Plasma-Treatable: {binary_counts[1]}", flush=True)

    all_results = []
    for model_name, builder_fn in MODEL_BUILDERS.items():
        result = run_kfold(model_name, builder_fn, paths, labels, DEVICE)
        all_results.append(result)

    make_plots(all_results, OUT_DIR)
    save_results_table(all_results, OUT_DIR)

    # Save JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {
        "experiment": "Cold Plasma Treatment Eligibility Classification",
        "clinical_context": "Binary classification of skin lesions for cold atmospheric plasma treatment planning",
        "dataset": "HAM10000 (balanced subset)",
        "num_images": len(paths), "num_classes": NUM_CLASSES,
        "binary_mapping": {
            "Benign (0)": ["benign_keratosis", "dermatofibroma", "melanocytic_nevi", "vascular_lesions"],
            "Plasma-Treatable (1)": ["melanoma", "basal_cell_carcinoma", "actinic_keratoses"],
        },
        "config": {"epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                   "k_folds": K_FOLDS, "img_size": IMG_SIZE, "max_per_class": MAX_PER_CLASS,
                   "optimizer": "AdamW", "scheduler": "CosineAnnealing",
                   "class_balancing": "WeightedRandomSampler + WeightedCrossEntropy"},
        "results": []
    }
    for r in all_results:
        summary["results"].append({
            "model": r["model"], "params": r["params"],
            "accuracy_mean": round(r["mean_acc"], 2), "accuracy_std": round(r["std_acc"], 2),
            "f1_mean": round(r["mean_f1"], 4), "f1_std": round(r["std_f1"], 4),
            "auc_mean": round(r["mean_auc"], 4), "auc_std": round(r["std_auc"], 4),
            "sensitivity_mean": round(r["mean_sensitivity"], 4), "sensitivity_std": round(r["std_sensitivity"], 4),
            "specificity_mean": round(r["mean_specificity"], 4), "specificity_std": round(r["std_specificity"], 4),
            "precision_mean": round(r["mean_precision"], 4), "precision_std": round(r["std_precision"], 4),
            "total_time_sec": round(r["total_time"], 1),
            "fold_accuracies": [round(fr["test_acc"],2) for fr in r["fold_results"]],
            "classification_report": r["overall_report"],
            "confusion_matrix": r["overall_cm"],
        })
    with open(f"{OUT_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final table
    print(f"\n{'='*100}", flush=True)
    print("  FINAL RESULTS - Cold Plasma Treatment Eligibility Classification", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Model':<25} {'Params':>10} {'Accuracy':>14} {'F1':>14} {'AUC':>14} {'Sensitivity':>14} {'Specificity':>14}", flush=True)
    print(f"{'-'*100}", flush=True)
    for r in all_results:
        print(f"{r['model']:<25} {r['params']:>10,} "
              f"{r['mean_acc']:>6.2f}+/-{r['std_acc']:<4.2f}% "
              f"{r['mean_f1']:>6.4f}+/-{r['std_f1']:<5.4f} "
              f"{r['mean_auc']:>6.4f}+/-{r['std_auc']:<5.4f} "
              f"{r['mean_sensitivity']:>6.4f}+/-{r['std_sensitivity']:<5.4f} "
              f"{r['mean_specificity']:>6.4f}+/-{r['std_specificity']:<5.4f}", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"\nAll results saved to {OUT_DIR}/", flush=True)
    print("Done!", flush=True)
