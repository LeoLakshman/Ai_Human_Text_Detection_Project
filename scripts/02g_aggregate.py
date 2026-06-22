import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

names = {"result_svm.json": "SVM", "result_dt.json": "Decision Tree",
         "result_ada.json": "AdaBoost", "result_fnn_sklearn.json": "FNN (MLP)"}
results = {}
for fname, label in names.items():
    with open(fname) as f:
        results[label] = json.load(f)

with open('classical_results.json', 'w') as f:
    json.dump(results, f, indent=2)

metrics_df = pd.DataFrame({k: {m: v[m] for m in ['accuracy', 'precision', 'recall', 'f1']}
                            for k, v in results.items()}).T
print(metrics_df)
metrics_df.to_csv('classical_metrics_table.csv')

fig, ax = plt.subplots(figsize=(7.5, 4.5))
metrics_df.plot(kind='bar', ax=ax, colormap='Set2')
ax.set_title('Model Comparison — Accuracy / Precision / Recall / F1')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('classical_model_comparison.png', dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, r in results.items():
    ax.plot(r['fpr'], r['tpr'], label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
plt.tight_layout()
plt.savefig('classical_roc_curves.png', dpi=130)
plt.close()

fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = np.array(r['confusion_matrix'])
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Human', 'AI'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Human', 'AI'])
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.savefig('classical_confusion_matrices.png', dpi=130)
plt.close()

print("Plots saved.")
