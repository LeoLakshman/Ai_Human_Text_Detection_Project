import time, json
import numpy as np
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

y_train = np.load('cache_y_train.npy')
y_test = np.load('cache_y_test.npy')
X_train_tfidf = load_npz('cache_X_train_tfidf.npz')
X_test_tfidf = load_npz('cache_X_test_tfidf.npz')
ling_train = np.load('cache_ling_train.npy')
ling_test = np.load('cache_ling_test.npy')
X_train_combo = load_npz('cache_X_train_combo.npz')
X_test_combo = load_npz('cache_X_test_combo.npz')

feature_sets = {
    "TF-IDF": (X_train_tfidf, X_test_tfidf),
    "Linguistic": (ling_train, ling_test),
    "TF-IDF+Linguistic": (X_train_combo, X_test_combo),
}

comparison = {}
for name, (Xtr, Xte) in feature_sets.items():
    t0 = time.time()
    clf = LinearSVC(C=1.0, max_iter=3000)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    comparison[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "train_time_sec": round(time.time() - t0, 2),
    }
    print(name, comparison[name])

with open('feature_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)
