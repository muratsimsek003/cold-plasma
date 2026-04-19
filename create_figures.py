import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

fig_dir = 'figures_final'
os.makedirs(fig_dir, exist_ok=True)

results = {
    'ViT-Tiny\n(scratch)': {
        'acc': 51.63, 'acc_std': 0.98, 'f1': 0.3622, 'f1_std': 0.2160,
        'auc': 0.5306, 'auc_std': 0.0076, 'sens': 0.3733, 'sens_std': 0.2450,
        'spec': 0.6469, 'spec_std': 0.2403, 'folds': [50.36, 52.74, 51.79],
        'train_acc': [52.8, 52.8, 52.7, 54.7, 55.9],
        'val_acc': [51.0, 50.2, 53.9, 54.6, 54.2],
        'train_loss': [0.7310, 0.6935, 0.6893, 0.6832, 0.6777],
        'val_loss': [0.6823, 0.6936, 0.6770, 0.6769, 0.6853],
        'cm': np.array([[425, 232], [377, 223]]),
        'color': '#C44E52', 'label': 'ViT-Tiny (scratch)',
    },
    'ViT-Tiny\n(pretrained)': {
        'acc': 77.88, 'acc_std': 0.79, 'f1': 0.7823, 'f1_std': 0.0117,
        'auc': 0.8677, 'auc_std': 0.0048, 'sens': 0.8333, 'sens_std': 0.0289,
        'spec': 0.7291, 'spec_std': 0.0143, 'folds': [77.33, 77.33, 79.00],
        'train_acc': [63.3, 72.9, 78.3, 82.6, 86.0],
        'val_acc': [71.1, 71.9, 77.2, 79.0, 82.6],
        'train_loss': [0.7274, 0.5220, 0.4470, 0.3535, 0.3261],
        'val_loss': [0.5685, 0.5153, 0.4501, 0.4579, 0.4307],
        'cm': np.array([[479, 178], [100, 500]]),
        'color': '#4C72B0', 'label': 'ViT-Tiny (pretrained)',
    },
    'ResNet-18\n(scratch)': {
        'acc': 61.81, 'acc_std': 1.91, 'f1': 0.6808, 'f1_std': 0.0127,
        'auc': 0.6780, 'auc_std': 0.0009, 'sens': 0.8575, 'sens_std': 0.0925,
        'spec': 0.3995, 'spec_std': 0.1210, 'folds': [59.90, 63.72],
        'train_acc': [58.4, 61.9, 60.4, 64.3, 65.9],
        'val_acc': [51.2, 60.9, 62.7, 65.1, 66.9],
        'train_loss': [0.6860, 0.6479, 0.6452, 0.6187, 0.6032],
        'val_loss': [0.8310, 0.6420, 0.6793, 0.6203, 0.6157],
        'cm': np.array([[264, 393], [97, 503]]),
        'color': '#55A868', 'label': 'ResNet-18 (scratch)',
    },
}
BN = ['Benign', 'Plasma-Treatable']

# Fig 8: Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for n, r in results.items():
    eps = range(1, 6)
    ax1.plot(eps, r['train_acc'], '-o', color=r['color'], label=r['label']+' (train)', markersize=5, lw=2)
    ax1.plot(eps, r['val_acc'], '--s', color=r['color'], label=r['label']+' (val)', markersize=5, lw=2, alpha=0.7)
    ax2.plot(eps, r['train_loss'], '-o', color=r['color'], label=r['label']+' (train)', markersize=5, lw=2)
    ax2.plot(eps, r['val_loss'], '--s', color=r['color'], label=r['label']+' (val)', markersize=5, lw=2, alpha=0.7)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)'); ax1.set_title('Training & Validation Accuracy')
ax1.legend(fontsize=7, loc='lower right'); ax1.grid(alpha=0.3); ax1.set_xticks(range(1,6))
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('Training & Validation Loss')
ax2.legend(fontsize=7); ax2.grid(alpha=0.3); ax2.set_xticks(range(1,6))
plt.tight_layout(); fig.savefig(f'{fig_dir}/Fig8_training_curves.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/Fig8_training_curves.pdf', bbox_inches='tight'); plt.close()
print('Fig8 done')

# Fig 9: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (n, r) in zip(axes, results.items()):
    cm = r['cm']; cm_n = cm.astype(float)/cm.sum(axis=1, keepdims=True)*100
    ax.imshow(cm_n, cmap=plt.cm.Blues, vmin=0, vmax=100)
    ax.set_title(f"{r['label']}\nAcc: {r['acc']:.1f}% | AUC: {r['auc']:.3f}", fontsize=10, fontweight='bold')
    ax.set_xticks(range(2)); ax.set_xticklabels(BN, fontsize=9)
    ax.set_yticks(range(2)); ax.set_yticklabels(BN, fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}\n({cm_n[i,j]:.1f}%)', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white' if cm_n[i,j]>50 else 'black')
plt.tight_layout(); fig.savefig(f'{fig_dir}/Fig9_confusion_matrices.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/Fig9_confusion_matrices.pdf', bbox_inches='tight'); plt.close()
print('Fig9 done')

# Fig 10: Model comparison
fig, ax = plt.subplots(figsize=(13, 6))
names = list(results.keys())
md = {
    'Accuracy': ([r['acc'] for r in results.values()], [r['acc_std'] for r in results.values()]),
    'Sensitivity': ([r['sens']*100 for r in results.values()], [r['sens_std']*100 for r in results.values()]),
    'Specificity': ([r['spec']*100 for r in results.values()], [r['spec_std']*100 for r in results.values()]),
    'AUC x100': ([r['auc']*100 for r in results.values()], [r['auc_std']*100 for r in results.values()]),
}
x = np.arange(len(names)); w = 0.2; clrs = ['#4C72B0','#DD8452','#55A868','#C44E52']
for i, (mn, (vals, stds)) in enumerate(md.items()):
    bars = ax.bar(x+(i-1.5)*w, vals, w, yerr=stds, capsize=4, label=mn, color=clrs[i], alpha=0.85, edgecolor='black', lw=0.5)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+3, f'{b.get_height():.1f}', ha='center', fontsize=7, fontweight='bold')
ax.set_ylabel('Score (%)'); ax.set_title('Model Performance Comparison (3-Fold CV)', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10); ax.legend(fontsize=9); ax.set_ylim(0,115); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); fig.savefig(f'{fig_dir}/Fig10_model_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/Fig10_model_comparison.pdf', bbox_inches='tight'); plt.close()
print('Fig10 done')

# Fig 11: Sensitivity vs Specificity
fig, ax = plt.subplots(figsize=(8, 6))
for n, r in results.items():
    ax.scatter(r['spec']*100, r['sens']*100, s=250, color=r['color'], marker='o', zorder=5, edgecolors='black', lw=1.5)
    ax.annotate(f"  {r['label']}\n  Acc={r['acc']:.1f}%", (r['spec']*100, r['sens']*100), fontsize=8, fontweight='bold')
ax.set_xlabel('Specificity (%)'); ax.set_ylabel('Sensitivity (%)')
ax.set_title('Sensitivity vs Specificity Trade-off', fontsize=12, fontweight='bold')
ax.set_xlim(20, 85); ax.set_ylim(25, 95)
ax.axhline(50, color='gray', ls='--', alpha=0.4); ax.axvline(50, color='gray', ls='--', alpha=0.4)
ax.fill_between([50,85], 50, 95, alpha=0.05, color='green')
ax.text(67, 55, 'Optimal\nRegion', fontsize=9, color='green', alpha=0.6, ha='center')
ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(f'{fig_dir}/Fig11_sens_spec.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/Fig11_sens_spec.pdf', bbox_inches='tight'); plt.close()
print('Fig11 done')

# Fig 12: Transfer learning impact
fig, ax = plt.subplots(figsize=(10, 5))
cats = ['Accuracy', 'F1-Score', 'AUC', 'Sensitivity', 'Specificity']
sc = [51.63, 36.22, 53.06, 37.33, 64.69]
pt = [77.88, 78.23, 86.77, 83.33, 72.91]
x = np.arange(len(cats)); w = 0.35
b1 = ax.bar(x-w/2, sc, w, label='ViT-Tiny (scratch)', color='#C44E52', alpha=0.85, edgecolor='black', lw=0.5)
b2 = ax.bar(x+w/2, pt, w, label='ViT-Tiny (pretrained)', color='#4C72B0', alpha=0.85, edgecolor='black', lw=0.5)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')
for i in range(len(cats)):
    ax.annotate(f'+{pt[i]-sc[i]:.1f}', xy=(x[i]+w/2, pt[i]+5), fontsize=7, color='green', fontweight='bold', ha='center')
ax.set_ylabel('Score (%)'); ax.set_title('Impact of Transfer Learning on ViT Performance', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10); ax.legend(fontsize=9); ax.set_ylim(0,105); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); fig.savefig(f'{fig_dir}/Fig12_transfer_learning.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/Fig12_transfer_learning.pdf', bbox_inches='tight'); plt.close()
print('Fig12 done')

# Fig 13: Boxplot
fig, ax = plt.subplots(figsize=(9, 5))
fa = [r['folds'] for r in results.values()]
bp = ax.boxplot(fa, tick_labels=[r['label'] for r in results.values()], patch_artist=True, widths=0.5)
for patch, r in zip(bp['boxes'], results.values()): patch.set_facecolor(r['color']); patch.set_alpha(0.6)
for med in bp['medians']: med.set_color('black'); med.set_linewidth(2)
for i, r in enumerate(results.values()):
    xp = np.random.normal(i+1, 0.04, len(r['folds']))
    ax.scatter(xp, r['folds'], color=r['color'], s=60, zorder=5, edgecolors='black', lw=0.8)
ax.set_ylabel('Accuracy (%)'); ax.set_title('Test Accuracy Distribution Across Folds', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y'); ax.axhline(50, color='red', ls='--', alpha=0.3, label='Random baseline')
ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f'{fig_dir}/Fig13_boxplot.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/Fig13_boxplot.pdf', bbox_inches='tight'); plt.close()
print('Fig13 done')

print(f'\nAll figures saved to {fig_dir}/')
for f in sorted(os.listdir(fig_dir)):
    print(f'  {f} ({os.path.getsize(os.path.join(fig_dir,f))//1024} KB)')
