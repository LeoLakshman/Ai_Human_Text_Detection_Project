"""
AI vs Human Text Detector — Streamlit App
Run with:  streamlit run app.py
"""
import os
import io
import json
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix

from utils.text_features import (extract_linguistic_features, LINGUISTIC_FEATURE_NAMES,
                                  simple_clean_text)

st.set_page_config(page_title="AI vs Human Text Detector", page_icon="🕵️", layout="wide")

MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_tfidf():
    path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_ling_scaler():
    path = os.path.join(MODELS_DIR, "linguistic_scaler.pkl")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_sklearn_models():
    """Load any of the classical/reference models that exist on disk."""
    candidates = {
        "SVM": "svm_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "AdaBoost": "adaboost_model.pkl",
        "FNN (sklearn MLP reference)": "fnn_sklearn_model.pkl",
    }
    loaded = {}
    for name, fname in candidates.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded


@st.cache_resource
def load_keras_models():
    """Load Keras .h5 models (FNN/LSTM/CNN). These were trained in Google Colab
    (see notebooks/project1_notebook.ipynb, Section 3.4-3.6) and are too large
    (~30MB each) to commit directly to the repo, so they're hosted as GitHub
    Release assets and downloaded here on first run, then cached in models/."""
    loaded = {}
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return loaded  # TensorFlow not installed in this environment

    from utils.remote_models import ensure_assets
    asset_names = ["fnn_model.h5", "lstm_model.h5", "cnn_model.h5"]
    download_status = ensure_assets(asset_names, _st=st)

    for name, fname in [("FNN (Keras)", "fnn_model.h5"),
                         ("LSTM", "lstm_model.h5"),
                         ("CNN", "cnn_model.h5")]:
        if not download_status.get(fname):
            continue  # not available locally or in the release — skip silently, listed in sidebar below
        path = os.path.join(MODELS_DIR, fname)
        try:
            loaded[name] = load_model(path)
        except Exception as e:
            st.sidebar.warning(f"Could not load {name}: {e}")
    return loaded


@st.cache_resource
def load_embedding_assets():
    """Load Word2Vec model + Keras tokenizer for the deep-learning models.
    Also hosted on the GitHub Release alongside the .h5 files (see load_keras_models)."""
    from utils.remote_models import ensure_assets
    ensure_assets(["word2vec.model", "tokenizer.json"], _st=st)

    w2v, tokenizer = None, None
    w2v_path = os.path.join(MODELS_DIR, "embedding_model", "word2vec.model")
    tok_path = os.path.join(MODELS_DIR, "embedding_model", "tokenizer.json")
    if os.path.exists(w2v_path):
        try:
            from gensim.models import Word2Vec
            w2v = Word2Vec.load(w2v_path)
        except ImportError:
            pass
    if os.path.exists(tok_path):
        try:
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            with open(tok_path) as f:
                tokenizer = tokenizer_from_json(f.read())
        except ImportError:
            pass
    return w2v, tokenizer


# ---------------------------------------------------------------------------
# Text extraction from uploads
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes):
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_bytes):
    import docx
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)


def extract_text(uploaded_file):
    raw = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(raw)
    elif uploaded_file.name.lower().endswith((".docx",)):
        return extract_text_from_docx(raw)
    else:
        return raw.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Feature building (must mirror the notebook's training pipeline exactly)
# ---------------------------------------------------------------------------
def build_classical_features(text, tfidf, ling_scaler):
    clean = simple_clean_text(text)
    X_tfidf = tfidf.transform([clean])
    ling = np.array([[extract_linguistic_features(clean)[name] for name in LINGUISTIC_FEATURE_NAMES]])
    ling_s = ling_scaler.transform(ling)
    return hstack([X_tfidf, csr_matrix(ling_s)]).tocsr(), ling[0]


def doc_vector(tokens, w2v_model, dim=100):
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def predict_with_model(name, text, models, tfidf, ling_scaler, w2v, tokenizer):
    """Returns (label_str, confidence_float_0to1)."""
    clean = simple_clean_text(text)

    if name in models["sklearn"]:
        model = models["sklearn"][name]
        X, _ = build_classical_features(text, tfidf, ling_scaler)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0, 1]
        else:
            proba = float(model.predict(X)[0])
        pred = int(proba >= 0.5)
        return pred, proba

    if name in models["keras"]:
        model = models["keras"][name]
        from utils.text_features import simple_tokenize
        if name == "FNN (Keras)":
            if w2v is None:
                raise RuntimeError("Word2Vec embeddings not found — train locally first.")
            vec = doc_vector(simple_tokenize(clean), w2v).reshape(1, -1)
            proba = float(model.predict(vec, verbose=0).ravel()[0])
        else:  # LSTM / CNN — need the tokenizer's padded sequence
            if tokenizer is None:
                raise RuntimeError("Tokenizer not found — train locally first.")
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([clean])
            padded = pad_sequences(seq, maxlen=250, padding="post", truncating="post")
            proba = float(model.predict(padded, verbose=0).ravel()[0])
        pred = int(proba >= 0.5)
        return pred, proba

    raise ValueError(f"Unknown model: {name}")


def explain_prediction(text, sklearn_models, tfidf, top_n=12):
    """Word-level explanation using the SVM's linear coefficients (most interpretable
    model here) applied to this document's TF-IDF weights."""
    if "SVM" not in sklearn_models or tfidf is None:
        return None
    svm = sklearn_models["SVM"]
    try:
        # CalibratedClassifierCV wraps one fitted LinearSVC per CV fold; average their coefs
        coefs = np.mean([cc.estimator.coef_[0] for cc in svm.calibrated_classifiers_], axis=0)
    except Exception:
        return None
    clean = simple_clean_text(text)
    X = tfidf.transform([clean])
    feature_names = np.array(tfidf.get_feature_names_out())
    nz = X.nonzero()[1]
    contributions = X[0, nz].toarray().ravel() * coefs[nz]
    order = np.argsort(-np.abs(contributions))[:top_n]
    return pd.DataFrame({
        "term": feature_names[nz][order],
        "contribution": contributions[order],
        "pushes_toward": np.where(contributions[order] > 0, "AI", "Human"),
    })


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🕵️ AI vs Human Text Detector")
st.caption("Upload a document or paste text, choose a model, and get a prediction with confidence and explanation.")

tfidf = load_tfidf()
ling_scaler = load_ling_scaler()
sklearn_models = load_sklearn_models()
keras_models = load_keras_models()
w2v, tokenizer = load_embedding_assets()
all_models = {"sklearn": sklearn_models, "keras": keras_models}
available_model_names = list(sklearn_models.keys()) + list(keras_models.keys())

if not available_model_names:
    st.error("No trained models found in `models/`. Run the notebook (`notebooks/project1_notebook.ipynb`) "
              "first to train and save the models.")
    st.stop()

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Provide text via:", ["Paste text", "Upload file (PDF/DOCX)"])
    text_input = ""
    if input_mode == "Paste text":
        text_input = st.text_area("Paste the text to analyze:", height=250,
                                    placeholder="Paste an essay, article, or any passage here...")
    else:
        uploaded = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
        if uploaded is not None:
            with st.spinner("Extracting text..."):
                text_input = extract_text(uploaded)
            st.success(f"Extracted {len(text_input.split())} words.")
            with st.expander("Preview extracted text"):
                st.write(text_input[:2000] + ("..." if len(text_input) > 2000 else ""))

    st.header("Model")
    model_choice = st.selectbox("Choose a classifier:", available_model_names)
    missing_dl = [n for n in ["FNN (Keras)", "LSTM", "CNN"] if n not in keras_models]
    if missing_dl:
        st.caption(f"Not loaded (train locally to enable): {', '.join(missing_dl)}")

    run_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

tab_predict, tab_compare, tab_report = st.tabs(["Prediction", "Model Comparison", "Downloadable Report"])

if run_btn and text_input.strip():
    word_count = len(text_input.split())
    if word_count < 20:
        st.warning("This text is quite short — predictions are more reliable on 50+ words.")

    # ---- Single-model prediction ----
    with tab_predict:
        try:
            pred, proba = predict_with_model(model_choice, text_input, all_models, tfidf, ling_scaler, w2v, tokenizer)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred, proba = None, None

        if pred is not None:
            label = "🤖 AI-Generated" if pred == 1 else "🧑 Human-Written"
            confidence = proba if pred == 1 else 1 - proba
            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("Prediction", label)
                st.metric("Confidence", f"{confidence*100:.1f}%")
                st.progress(float(confidence))
            with c2:
                ling_feats = extract_linguistic_features(simple_clean_text(text_input))
                st.write("**Quick stylistic signals**")
                st.write(f"- Words: {word_count}")
                st.write(f"- Avg sentence length: {ling_feats['avg_sentence_length']:.1f} words")
                st.write(f"- Vocabulary richness (TTR): {ling_feats['type_token_ratio']:.3f}")
                st.write(f"- Flesch Reading Ease: {ling_feats['flesch_reading_ease']:.1f}")
                st.write(f"- Contraction use: {ling_feats['contraction_ratio']:.4f}")

            st.subheader("Why this prediction? (word-level explanation)")
            exp_df = explain_prediction(text_input, sklearn_models, tfidf)
            if exp_df is not None and len(exp_df):
                fig, ax = plt.subplots(figsize=(7, 4))
                colors = exp_df["contribution"].apply(lambda v: "#DD8452" if v > 0 else "#4C72B0")
                ax.barh(exp_df["term"][::-1], exp_df["contribution"][::-1], color=colors[::-1])
                ax.set_xlabel("Contribution (← Human  |  AI →)")
                st.pyplot(fig)
                st.caption("Explanation derived from the SVM's linear weights × this document's TF-IDF values "
                            "(shown for any selected model, since SVM is the most directly interpretable).")
            else:
                st.info("Word-level explanation requires the SVM model to be available.")

    # ---- Side-by-side model comparison ----
    with tab_compare:
        st.subheader("All available models on this same text")
        rows = []
        for name in available_model_names:
            try:
                p, pr = predict_with_model(name, text_input, all_models, tfidf, ling_scaler, w2v, tokenizer)
                conf = pr if p == 1 else 1 - pr
                rows.append({"Model": name, "Prediction": "AI" if p == 1 else "Human",
                              "Confidence": f"{conf*100:.1f}%", "P(AI)": round(float(pr), 4)})
            except Exception as e:
                rows.append({"Model": name, "Prediction": "error", "Confidence": "-", "P(AI)": str(e)})
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        plot_df = comp_df[comp_df["P(AI)"].apply(lambda x: isinstance(x, float))]
        if len(plot_df):
            ax.bar(plot_df["Model"], plot_df["P(AI)"], color="#DD8452")
            ax.axhline(0.5, color="gray", linestyle="--")
            ax.set_ylabel("P(AI-generated)")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=15)
            st.pyplot(fig)

    # ---- Downloadable report ----
    with tab_report:
        st.subheader("Generate a downloadable analysis report")
        report_lines = [
            "AI vs HUMAN TEXT DETECTION REPORT",
            f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"Word count: {word_count}",
            "",
            f"Selected model: {model_choice}",
            f"Prediction: {'AI-Generated' if pred == 1 else 'Human-Written' if pred == 0 else 'N/A'}",
            f"Confidence: {confidence*100:.1f}%" if pred is not None else "",
            "",
            "All-model comparison:",
        ]
        for r in rows:
            report_lines.append(f"  - {r['Model']}: {r['Prediction']} ({r['Confidence']})")
        report_lines += ["", "Stylistic signals:"]
        for k, v in ling_feats.items():
            report_lines.append(f"  - {k}: {v:.4f}")
        report_text = "\n".join(report_lines)
        st.text_area("Report preview", report_text, height=300)
        st.download_button("⬇️ Download report (.txt)", report_text,
                             file_name="ai_human_detection_report.txt")
else:
    with tab_predict:
        st.info("Paste text or upload a file, choose a model in the sidebar, then click **Analyze**.")
