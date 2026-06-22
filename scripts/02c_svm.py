import time, json, joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)

y_train = np.load('cache_y_train.npy')
y_test = np.load('cache_y_test.npy')
X_train = load_npz('cache_X_train_combo.npz')
X_test = load_npz('cache_X_test_combo.npz')

t0 = time.time()
grid = {"C": [0.01, 0.1, 1, 10]}
search = GridSearchCV(LinearSVC(max_iter=4000), grid, cv=3, scoring="f1", n_jobs=1)
search.fit(X_train, y_train)
print("SVM best params:", search.best_params_, f"({time.time()-t0:.1f}s)")

# wrap with calibration so we get probability scores for the app's confidence display
best_svm = CalibratedClassifierCV(LinearSVC(C=search.best_params_["C"], max_iter=4000), cv=3)
best_svm.fit(X_train, y_train)

pred = best_svm.predict(X_test)
proba = best_svm.predict_proba(X_test)[:, 1]
acc, prec, rec, f1 = (accuracy_score(y_test, pred), precision_score(y_test, pred),
                       recall_score(y_test, pred), f1_score(y_test, pred))
cm = confusion_matrix(y_test, pred)
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)
print(classification_report(y_test, pred, target_names=['Human', 'AI']))

result = {"best_params": search.best_params_, "accuracy": acc, "precision": prec,
          "recall": rec, "f1": f1, "roc_auc": roc_auc, "confusion_matrix": cm.tolist(),
          "fpr": fpr.tolist(), "tpr": tpr.tolist()}
with open('result_svm.json', 'w') as f:
    json.dump(result, f, indent=2)
joblib.dump(best_svm, 'models/svm_model.pkl')
print("Saved SVM model + results.")
