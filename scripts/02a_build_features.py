import sys, joblib, time
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, save_npz
from utils.text_features import extract_linguistic_matrix

train_df = pd.read_csv('data/training_data/train.csv')
test_df = pd.read_csv('data/test_data/test.csv')

t0 = time.time()
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 1), min_df=5, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df['text_clean'])
X_test_tfidf = tfidf.transform(test_df['text_clean'])
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("tfidf", time.time() - t0, X_train_tfidf.shape)

t0 = time.time()
ling_train = extract_linguistic_matrix(train_df['text_clean'].tolist())
ling_test = extract_linguistic_matrix(test_df['text_clean'].tolist())
ling_scaler = StandardScaler().fit(ling_train)
ling_train_s = ling_scaler.transform(ling_train)
ling_test_s = ling_scaler.transform(ling_test)
joblib.dump(ling_scaler, 'models/linguistic_scaler.pkl')
print("linguistic", time.time() - t0, ling_train_s.shape)

X_train_combo = hstack([X_train_tfidf, csr_matrix(ling_train_s)]).tocsr()
X_test_combo = hstack([X_test_tfidf, csr_matrix(ling_test_s)]).tocsr()

save_npz('cache_X_train_tfidf.npz', X_train_tfidf)
save_npz('cache_X_test_tfidf.npz', X_test_tfidf)
save_npz('cache_X_train_combo.npz', X_train_combo)
save_npz('cache_X_test_combo.npz', X_test_combo)
np.save('cache_ling_train.npy', ling_train_s)
np.save('cache_ling_test.npy', ling_test_s)
np.save('cache_y_train.npy', train_df['label'].values)
np.save('cache_y_test.npy', test_df['label'].values)
print("Saved all feature caches.")
