import time, json, joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)

y_train = np.load('cache_y_train.npy')
y_test = np.load('cache_y_test.npy')
X_train = load_npz('cache_X_train_combo.npz')
X_test = load_npz('cache_X_test_combo.npz')

# Tune on a stratified subsample for speed (single-core sandbox), then refit best params on full data
rng = np.random.RandomState(42)
idx0 = np.where(y_train == 0)[0]; idx1 = np.where(y_train == 1)[0]
sub_idx = np.concatenate([rng.choice(idx0, 1500, replace=False), rng.choice(idx1, 1500, replace=False)])
X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]

t0 = time.time()
grid = {"max_depth": [15, 30, None], "min_samples_split": [2, 10]}
search = GridSearchCV(DecisionTreeClassifier(random_state=42), grid, cv=2, scoring="f1", n_jobs=1)
search.fit(X_sub, y_sub)
print("DT best params (from subsample search):", search.best_params_, f"({time.time()-t0:.1f}s)")

best_dt = DecisionTreeClassifier(random_state=42, **search.best_params_)
best_dt.fit(X_train, y_train)

pred = best_dt.predict(X_test)
proba = best_dt.predict_proba(X_test)[:, 1]
acc, prec, rec, f1 = (accuracy_score(y_test, pred), precision_score(y_test, pred),
                       recall_score(y_test, pred), f1_score(y_test, pred))
cm = confusion_matrix(y_test, pred)
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
print(classification_report(y_test, pred, target_names=['Human', 'AI']))

result = {"best_params": search.best_params_, "accuracy": acc, "precision": prec,
          "recall": rec, "f1": f1, "roc_auc": roc_auc, "confusion_matrix": cm.tolist(),
          "fpr": fpr.tolist(), "tpr": tpr.tolist()}
with open('result_dt.json', 'w') as f:
    json.dump(result, f, indent=2)
joblib.dump(best_dt, 'models/decision_tree_model.pkl')
print("Saved Decision Tree model + results.")
