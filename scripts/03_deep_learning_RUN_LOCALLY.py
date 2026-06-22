"""
Deep Learning models + Word Embeddings — RUN THIS LOCALLY.

This script was written and carefully checked in the project sandbox, but NOT
executed there: that sandbox has no internet access, so tensorflow/gensim
cannot be installed in it. Run this on your own machine (or Google Colab)
after `pip install -r requirements.txt`. It will create:
    models/embedding_model/word2vec.model
    models/fnn_model.h5
    models/lstm_model.h5
    models/cnn_model.h5
    deep_learning_results.json   (metrics, for the notebook's comparison section)
    dl_*.png                     (training curves / ROC, for the notebook)

It expects data/training_data/train.csv and data/test_data/test.csv to already
exist (created by 01_eda.py / the notebook's Section 1).
"""

import sys, json, time
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gensim.models import Word2Vec

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout, Conv1D,
                                      GlobalMaxPooling1D, Input, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)

from utils.text_features import simple_tokenize

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
train_df = pd.read_csv('data/training_data/train.csv')
test_df = pd.read_csv('data/test_data/test.csv')
y_train = train_df['label'].values
y_test = test_df['label'].values

train_tokens = [simple_tokenize(t) for t in train_df['text_clean']]
test_tokens = [simple_tokenize(t) for t in test_df['text_clean']]

MAX_VOCAB = 15000
MAX_LEN = 250          # truncate/pad sequences to this many tokens
EMBED_DIM = 100

# ------------------------------------------------------------------
# 2. Word embeddings — Word2Vec trained on this corpus (gensim)
#    (You can swap in pretrained GloVe vectors instead; see note below.)
# ------------------------------------------------------------------
print("Training Word2Vec...")
w2v = Word2Vec(sentences=train_tokens, vector_size=EMBED_DIM, window=5,
                min_count=2, workers=4, epochs=10, seed=42)
import os
os.makedirs('models/embedding_model', exist_ok=True)
w2v.save('models/embedding_model/word2vec.model')
print("Word2Vec vocab size:", len(w2v.wv))

# Quick sanity check of the embedding space (qualitative — put a few of these
# in the notebook's write-up under "Feature Engineering")
for probe in ["said", "however", "data"]:
    if probe in w2v.wv:
        print(probe, "->", w2v.wv.most_similar(probe, topn=5))

# NOTE — using pretrained GloVe instead of training your own Word2Vec:
#   1. Download glove.6B.100d.txt from https://nlp.stanford.edu/projects/glove/
#   2. Parse it into a dict {word: vector}, then build the embedding_matrix
#      below from that dict instead of from `w2v.wv`.

# ------------------------------------------------------------------
# 3. Tokenizer + padded integer sequences (for the Keras Embedding layer)
# ------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['text_clean'])
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_df['text_clean']),
                             maxlen=MAX_LEN, padding='post', truncating='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test_df['text_clean']),
                            maxlen=MAX_LEN, padding='post', truncating='post')

vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

# Build an embedding matrix initialized from our trained Word2Vec vectors
embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, EMBED_DIM)).astype('float32')
hits = 0
for word, idx in tokenizer.word_index.items():
    if idx >= vocab_size:
        continue
    if word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word]
        hits += 1
print(f"Embedding matrix coverage: {hits}/{vocab_size} words from Word2Vec")

with open('models/embedding_model/tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())

# ------------------------------------------------------------------
# Helper: train + evaluate + plot any Keras model
# ------------------------------------------------------------------
dl_results = {}

def train_and_eval(name, model, X_tr, X_te, epochs=8, batch_size=32):
    t0 = time.time()
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(X_tr, y_train, validation_split=0.15, epochs=epochs,
                         batch_size=batch_size, callbacks=[es], verbose=2)
    train_time = time.time() - t0

    proba = model.predict(X_te, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    acc, prec, rec, f1 = (accuracy_score(y_test, pred), precision_score(y_test, pred),
                           recall_score(y_test, pred), f1_score(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    print(f"\n=== {name} ===  (trained in {train_time:.1f}s)")
    print(classification_report(y_test, pred, target_names=['Human', 'AI']))

    dl_results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                         "roc_auc": roc_auc, "confusion_matrix": cm.tolist(),
                         "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                         "train_time_sec": train_time}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history.history['loss'], label='train loss')
    axes[0].plot(history.history['val_loss'], label='val loss')
    axes[0].set_title(f'{name} — Loss'); axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='train acc')
    axes[1].plot(history.history['val_accuracy'], label='val acc')
    axes[1].set_title(f'{name} — Accuracy'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"dl_{name.lower().replace(' ', '_')}_training_curves.png", dpi=130)
    plt.close()
    return model

# ------------------------------------------------------------------
# 4. FNN — Feedforward Neural Network
#    Input: mean-pooled Word2Vec embedding per document (a fixed-size vector)
# ------------------------------------------------------------------
def doc_vector(tokens, w2v_model, dim=EMBED_DIM):
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

X_train_fnn = np.array([doc_vector(t, w2v) for t in train_tokens])
X_test_fnn = np.array([doc_vector(t, w2v) for t in test_tokens])

fnn = Sequential([
    Input(shape=(EMBED_DIM,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])
fnn.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
fnn.summary()
fnn = train_and_eval("FNN", fnn, X_train_fnn, X_test_fnn, epochs=20, batch_size=32)
fnn.save('models/fnn_model.h5')

# ------------------------------------------------------------------
# 5. LSTM
# ------------------------------------------------------------------
lstm = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(vocab_size, EMBED_DIM, weights=[embedding_matrix], trainable=True),
    Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])
lstm.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
lstm.summary()
lstm = train_and_eval("LSTM", lstm, X_train_seq, X_test_seq, epochs=8, batch_size=32)
lstm.save('models/lstm_model.h5')

# ------------------------------------------------------------------
# 6. CNN for Text
# ------------------------------------------------------------------
cnn = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(vocab_size, EMBED_DIM, weights=[embedding_matrix], trainable=True),
    Conv1D(128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])
cnn.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
cnn.summary()
cnn = train_and_eval("CNN", cnn, X_train_seq, X_test_seq, epochs=8, batch_size=32)
cnn.save('models/cnn_model.h5')

# ------------------------------------------------------------------
# 7. Save aggregate results + comparison plots
# ------------------------------------------------------------------
with open('deep_learning_results.json', 'w') as f:
    json.dump(dl_results, f, indent=2)

metrics_df = pd.DataFrame({k: {m: v[m] for m in ['accuracy', 'precision', 'recall', 'f1']}
                            for k, v in dl_results.items()}).T
print(metrics_df)

fig, ax = plt.subplots(figsize=(7, 4.5))
metrics_df.plot(kind='bar', ax=ax, colormap='Set2')
ax.set_title('Deep Learning Model Comparison'); ax.set_ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('dl_model_comparison.png', dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, r in dl_results.items():
    ax.plot(r['fpr'], r['tpr'], label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Deep Learning Models'); ax.legend()
plt.tight_layout()
plt.savefig('dl_roc_curves.png', dpi=130)
plt.close()

print("\nDONE. Deep learning models, embeddings, metrics, and plots saved.")
