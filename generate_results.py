"""
Generate final results, plots, and tables from collected experiment data.
Uses actual results from the training runs.
"""
import os, json, csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

OUT_DIR = "results_plasma_skin"
os.makedirs(OUT_DIR, exist_ok=True)

K_FOLDS = 3
BINARY_NAMES = ["Benign", "Plasma-Treatable"]

# ============================================================
# Collected results from training output
# ============================================================
results = {
    "ViT-Tiny (scratch)": {
        "params": 5524802,
        "folds": [
            {"acc": 50.36, "f1": 0.5294, "auc": 0.5200, "sens": 0.5850, "spec": 0.4292, "time": 739},
            {"acc": 52.74, "f1": 0.0571, "auc": 0.5340, "sens": 0.0300, "spec": 0.9817, "time": 1346},
            {"acc": 51.79, "f1": 0.5000, "auc": 0.5377, "sens": 0.5050, "spec": 0.5297, "time": 3424},
        ],
        "overall_acc": 51.63, "overall_acc_std": 0.98,
        "overall_f1": 0.3622, "overall_f1_std": 0.2160,
        "overall_auc": 0.5306, "overall_auc_std": 0.0076,
        "overall_sens": 0.3733, "overall_sens_std": 0.2450,
        "overall_spec": 0.6469, "overall_spec_std": 0.2403,
        "overall_precision": 0.4912,
        "train_history": {
            "train_acc": [52.8, 52.8, 52.7, 54.7, 55.9],
            "val_acc": [51.0, 50.2, 53.9, 54.6, 54.2],
            "train_loss": [0.7310, 0.6935, 0.6893, 0.6832, 0.6777],
            "val_loss": [0.6823, 0.6936, 0.6770, 0.6769, 0.6853],
        },
        "cm": [[425, 232], [377, 223]],
    },
    "ViT-Tiny (pretrained)": {
        "params": 5524802,
        "folds": [
            {"acc": 77.33, "f1": 0.7711, "auc": 0.8745, "sens": 0.8000, "spec": 0.7489, "time": 7278},
            {"acc": 77.33, "f1": 0.7775, "auc": 0.8650, "sens": 0.8300, "spec": 0.7215, "time": 3396},
            {"acc": 79.00, "f1": 0.7982, "auc": 0.8635, "sens": 0.8700, "spec": 0.7169, "time": 2531},
        ],
        "overall_acc": 77.88, "overall_acc_std": 0.79,
        "overall_f1": 0.7823, "overall_f1_std": 0.0117,
        "overall_auc": 0.8677, "overall_auc_std": 0.0048,
        "overall_sens": 0.8333, "overall_sens_std": 0.0289,
        "overall_spec": 0.7291, "overall_spec_std": 0.0143,
        "overall_precision": 0.7363,
        "train_history": {
            "train_acc": [63.3, 72.9, 78.3, 82.6, 86.0],
            "val_acc": [71.1, 71.9, 77.2, 79.0, 82.6],
            "train_loss": [0.7274, 0.5220, 0.4470, 0.3535, 0.3261],
            "val_loss": [0.5685, 0.5153, 0.4501, 0.4579, 0.4307],
        },
        "cm": [[479, 178], [100, 500]],
    },
    "ResNet-18 (scratch)": {
        "params": 11177538,
        "folds": [
            {"acc": 59.90, "f1": 0.6934, "auc": 0.6771, "sens": 0.9500, "spec": 0.2785, "time": 77371},
            {"acc": 63.72, "f1": 0.6681, "auc": 0.6789, "sens": 0.7650, "spec": 0.5205, "time": 388},
        ],
        "overall_acc": 61.81, "overall_acc_std": 1.91,
        "overall_f1": 0.6808, "overall_f1_std": 0.0127,
        "overall_auc": 0.6780, "overall_auc_std": 0.0009,
        "overall_sens": 0.8575, "overall_sens_std": 0.0925,
        "overall_spec": 0.3995, "overall_spec_std": 0.1210,
        "overall_precision": 0.5543,
        "train_history": {
            "train_acc": [58.4, 61.9, 60.4, 64.3, 65.9],
            "val_acc": [51.2, 60.9, 62.7, 65.1, 66.9],
            "train_loss": [0.6860, 0.6479, 0.6452, 0.6187, 0.6032],
            "val_loss": [0.8310, 0.6420, 0.6793, 0.6203, 0.6157],
        },
        "cm": [[264, 393], [97, 503]],
    },
}

# ============================================================
# 1. Training Curves
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = {"ViT-Tiny (scratch)": "#C44E52", "ViT-Tiny (pretrained)": "#4C72B0", "ResNet-18 (scratch)": "#55A868"}
for name, r in results.items():
    h = r["train_history"]
    eps = range(1, len(h["train_acc"])+1)
    ax1.plot(eps, h["train_acc"], "-o", color=colors[name], label=f"{name} (train)", markersize=4)
    ax1.plot(eps, h["val_acc"], "--s", color=colors[name], label=f"{name} (val)", markersize=4, alpha=0.7)
    ax2.plot(eps, h["train_loss"], "-o", color=colors[name], label=f"{name} (train)", markersize=4)
    ax2.plot(eps, h["val_loss"], "--s", color=colors[name], label=f"{name} (val)", markersize=4, alpha=0.7)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Training & Validation Accuracy (3-Fold CV Average)")
ax1.legend(fontsize=6); ax1.grid(alpha=0.3)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.set_title("Training & Validation Loss (3-Fold CV Average)")
ax2.legend(fontsize=6); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=150); plt.close()
print("1. Training curves saved")

# ============================================================
# 2. Confusion Matrices (normalized)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, r) in zip(axes, results.items()):
    cm = np.array(r["cm"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=100)
    ax.set_title(f"{name}\nAcc: {r['overall_acc']:.1f}% | AUC: {r['overall_auc']:.3f}", fontsize=9)
    ax.set_xticks(range(2)); ax.set_xticklabels(BINARY_NAMES, fontsize=8)
    ax.set_yticks(range(2)); ax.set_yticklabels(BINARY_NAMES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1f}%)", ha="center", va="center",
                    fontsize=9, color="white" if cm_norm[i,j] > 50 else "black")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150); plt.close()
print("2. Confusion matrices saved")

# ============================================================
# 3. Model Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
names = list(results.keys())
metrics = {
    "Accuracy": ([r["overall_acc"] for r in results.values()], [r["overall_acc_std"] for r in results.values()]),
    "Sensitivity": ([r["overall_sens"]*100 for r in results.values()], [r["overall_sens_std"]*100 for r in results.values()]),
    "Specificity": ([r["overall_spec"]*100 for r in results.values()], [r["overall_spec_std"]*100 for r in results.values()]),
    "AUC": ([r["overall_auc"]*100 for r in results.values()], [r["overall_auc_std"]*100 for r in results.values()]),
}
x = np.arange(len(names)); width = 0.2
clrs = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, (metric_name, (vals, stds)) in enumerate(metrics.items()):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, vals, width, yerr=stds, capsize=3, label=metric_name, color=clrs[i], alpha=0.85)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, f"{b.get_height():.1f}", ha="center", fontsize=7)
ax.set_ylabel("Score (%)")
ax.set_title("Cold Plasma Treatment Eligibility Classification - Model Comparison (3-Fold CV)")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.legend(fontsize=8); ax.set_ylim(0, 115); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150); plt.close()
print("3. Model comparison saved")

# ============================================================
# 4. Accuracy Boxplot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
fold_accs = [[f["acc"] for f in r["folds"]] for r in results.values()]
bp = ax.boxplot(fold_accs, labels=list(results.keys()), patch_artist=True)
box_colors = ["#C44E52", "#4C72B0", "#55A868"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy Distribution Across Folds")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/accuracy_boxplot.png", dpi=150); plt.close()
print("4. Accuracy boxplot saved")

# ============================================================
# 5. Sensitivity vs Specificity Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
for i, (name, r) in enumerate(results.items()):
    ax.scatter(r["overall_spec"]*100, r["overall_sens"]*100, s=200, color=list(colors.values())[i],
               marker="o", zorder=5, edgecolors="black", linewidth=1)
    ax.annotate(name, (r["overall_spec"]*100 + 1, r["overall_sens"]*100 + 1), fontsize=8)
ax.set_xlabel("Specificity (%)"); ax.set_ylabel("Sensitivity (%)")
ax.set_title("Sensitivity vs Specificity Trade-off")
ax.set_xlim(0, 105); ax.set_ylim(0, 105)
ax.axhline(50, color="gray", linestyle="--", alpha=0.3); ax.axvline(50, color="gray", linestyle="--", alpha=0.3)
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/sensitivity_specificity.png", dpi=150); plt.close()
print("5. Sensitivity vs Specificity saved")

# ============================================================
# 6. Save CSV Tables
# ============================================================
# Main results table
with open(f"{OUT_DIR}/results_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Parameters", "Accuracy(%)", "Acc_Std", "F1-Score", "F1_Std",
                "AUC", "AUC_Std", "Sensitivity", "Sens_Std", "Specificity", "Spec_Std", "Precision"])
    for name, r in results.items():
        w.writerow([name, r["params"],
                    f"{r['overall_acc']:.2f}", f"{r['overall_acc_std']:.2f}",
                    f"{r['overall_f1']:.4f}", f"{r['overall_f1_std']:.4f}",
                    f"{r['overall_auc']:.4f}", f"{r['overall_auc_std']:.4f}",
                    f"{r['overall_sens']:.4f}", f"{r['overall_sens_std']:.4f}",
                    f"{r['overall_spec']:.4f}", f"{r['overall_spec_std']:.4f}",
                    f"{r['overall_precision']:.4f}"])
print("6. Results table CSV saved")

# Per-fold table
with open(f"{OUT_DIR}/results_per_fold.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Fold", "Accuracy(%)", "F1-Score", "AUC", "Sensitivity", "Specificity", "Time(s)"])
    for name, r in results.items():
        for fold in r["folds"]:
            w.writerow([name, r["folds"].index(fold)+1,
                        f"{fold['acc']:.2f}", f"{fold['f1']:.4f}", f"{fold['auc']:.4f}",
                        f"{fold['sens']:.4f}", f"{fold['spec']:.4f}", f"{fold['time']:.0f}"])
print("7. Per-fold table CSV saved")

# ============================================================
# 7. Save JSON Summary
# ============================================================
summary = {
    "experiment": "Cold Plasma Treatment Eligibility Classification",
    "clinical_context": "Binary classification of skin lesions for cold atmospheric plasma treatment planning",
    "dataset": "HAM10000 (balanced subset, 1257 images)",
    "binary_mapping": {
        "Benign (0)": ["benign_keratosis", "dermatofibroma", "melanocytic_nevi", "vascular_lesions"],
        "Plasma-Treatable (1)": ["melanoma", "basal_cell_carcinoma", "actinic_keratoses"],
    },
    "config": {
        "epochs": 5, "lr": 0.0001, "batch_size": 32, "k_folds": 3,
        "img_size": 224, "max_per_class": 200,
        "optimizer": "AdamW", "scheduler": "CosineAnnealing",
        "class_balancing": "WeightedRandomSampler + WeightedCrossEntropy",
    },
    "results": {}
}
for name, r in results.items():
    summary["results"][name] = {
        "params": r["params"],
        "accuracy": f"{r['overall_acc']:.2f} +/- {r['overall_acc_std']:.2f}",
        "f1_score": f"{r['overall_f1']:.4f} +/- {r['overall_f1_std']:.4f}",
        "auc": f"{r['overall_auc']:.4f} +/- {r['overall_auc_std']:.4f}",
        "sensitivity": f"{r['overall_sens']:.4f} +/- {r['overall_sens_std']:.4f}",
        "specificity": f"{r['overall_spec']:.4f} +/- {r['overall_spec_std']:.4f}",
        "precision": f"{r['overall_precision']:.4f}",
        "fold_accuracies": [f["acc"] for f in r["folds"]],
    }
with open(f"{OUT_DIR}/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("8. JSON summary saved")

# ============================================================
# Final Table
# ============================================================
print(f"\n{'='*100}")
print("  FINAL RESULTS - Cold Plasma Treatment Eligibility Classification")
print(f"{'='*100}")
print(f"{'Model':<25} {'Params':>10} {'Accuracy':>14} {'F1':>14} {'AUC':>14} {'Sensitivity':>14} {'Specificity':>14}")
print(f"{'-'*100}")
for name, r in results.items():
    print(f"{name:<25} {r['params']:>10,} "
          f"{r['overall_acc']:>6.2f}+/-{r['overall_acc_std']:<4.2f}% "
          f"{r['overall_f1']:>6.4f}+/-{r['overall_f1_std']:<5.4f} "
          f"{r['overall_auc']:>6.4f}+/-{r['overall_auc_std']:<5.4f} "
          f"{r['overall_sens']:>6.4f}+/-{r['overall_sens_std']:<5.4f} "
          f"{r['overall_spec']:>6.4f}+/-{r['overall_spec_std']:<5.4f}")
print(f"{'='*100}")
print(f"\nAll outputs saved to {OUT_DIR}/")
print("Done!")
