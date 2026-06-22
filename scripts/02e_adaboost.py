import time, json, joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)

y_train = np.load('cache_y_train.npy')
y_test = np.load('cache_y_test.npy')
X_train = load_npz('cache_X_train_combo.npz')
X_test = load_npz('cache_X_test_combo.npz')

rng = np.random.RandomState(42)
idx0 = np.where(y_train == 0)[0]; idx1 = np.where(y_train == 1)[0]
sub_idx = np.concatenate([rng.choice(idx0, 1500, replace=False), rng.choice(idx1, 1500, replace=False)])
X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]

t0 = time.time()
grid = {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}
search = GridSearchCV(AdaBoostClassifier(random_state=42), grid, cv=2, scoring="f1", n_jobs=1)
search.fit(X_sub, y_sub)
print("AdaBoost best params (from subsample search):", search.best_params_, f"({time.time()-t0:.1f}s)")

best_ada = AdaBoostClassifier(random_state=42, **search.best_params_)
t0 = time.time()
best_ada.fit(X_train, y_train)
print(f"refit on full train: {time.time()-t0:.1f}s")

pred = best_ada.predict(X_test)
proba = best_ada.predict_proba(X_test)[:, 1]
acc, prec, rec, f1 = (accuracy_score(y_test, pred), precision_score(y_test, pred),
                       recall_score(y_test, pred), f1_score(y_test, pred))
cm = confusion_matrix(y_test, pred)
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
print(classification_report(y_test, pred, target_names=['Human', 'AI']))

result = {"best_params": search.best_params_, "accuracy": acc, "precision": prec,
          "recall": rec, "f1": f1, "roc_auc": roc_auc, "confusion_matrix": cm.tolist(),
          "fpr": fpr.tolist(), "tpr": tpr.tolist()}
with open('result_ada.json', 'w') as f:
    json.dump(result, f, indent=2)
joblib.dump(best_ada, 'models/adaboost_model.pkl')
print("Saved AdaBoost model + results.")
