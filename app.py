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
    """Load your high-confidence Keras .h5 models (FNN/LSTM/CNN) directly from disk."""
    loaded = {}
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return loaded  # TensorFlow not installed in this environment
        
    for name, fname in [("FNN (Keras)", "fnn_model.h5"),
                        ("LSTM", "lstm_model.h5"),
                        ("CNN", "cnn_model.h5")]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                loaded[name] = load_model(path, compile=False)
            except Exception as e:
                st.sidebar.warning(f"Could not load {name}: {e}")
    return loaded


@st.cache_resource
def load_embedding_assets():
    """Load the Keras text tokenizer sequence asset for deep-learning models."""
    tokenizer = None
    tok_candidates = [
        os.path.join(MODELS_DIR, "tokenizer.json"),
        os.path.join(MODELS_DIR, "embedding_model", "tokenizer.json")
    ]
    
    for path in tok_candidates:
        if os.path.exists(path):
            try:
                from tensorflow.keras.preprocessing.text import tokenizer_from_json
                with open(path, 'r', encoding='utf-8') as f:
                    tokenizer = tokenizer_from_json(f.read())
                break
            except Exception:
                pass
    return tokenizer


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


def predict_with_model(name, text, models, tfidf, ling_scaler, tokenizer):
    """Returns (pred_label, confidence_float)."""
    clean = simple_clean_text(text)

    # Traditional ML Model routing logic
    if name in models["sklearn"]:
        model = models["sklearn"][name]
        X, _ = build_classical_features(text, tfidf, ling_scaler)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0, 1]
        else:
            proba = float(model.predict(X)[0])
        pred = int(proba >= 0.5)
        return pred, proba

    # Keras Neural Networks parsing routing matrix logic
    if name in models["keras"]:
        model = models["keras"][name]
        
        if tokenizer is None:
            raise RuntimeError("Sequence tokenizer asset 'tokenizer.json' not found in models/ folder. Please run your Colab notebook export block and upload it.")
            
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=250, padding="post", truncating="post")
        
        proba = float(model.predict(padded, verbose=0).ravel()[0])
        pred = int(proba >= 0.5)
        return pred, proba

    raise ValueError(f"Unknown model architecture descriptor parameter: {name}")


def explain_prediction(text, sklearn_models, tfidf, top_n=12):
    """Word-level explanation using the SVM's linear coefficients applied to this document's TF-IDF weights."""
    if "SVM" not in sklearn_models or tfidf is None:
        return None
    svm = sklearn_models["SVM"]
    try:
        if hasattr(svm, "calibrated_classifiers_"):
            coefs = np.mean([cc.estimator.coef_[0] for cc in svm.calibrated_classifiers_], axis=0)
        else:
            coefs = svm.coef_[0]
            if hasattr(coefs, "toarray"):
                coefs = coefs.toarray().ravel()
    except Exception:
        return None
        
    clean = simple_clean_text(text)
    X = tfidf.transform([clean])
    feature_names = np.array(tfidf.get_feature_names_out())
    nz = X.nonzero()[1]
    
    if len(nz) == 0:
        return None
        
    contributions = X[0, nz].toarray().ravel() * coefs[nz]
    order = np.argsort(-np.abs(contributions))[:top_n]
    return pd.DataFrame({
        "term": feature_names[nz][order],
        "contribution": contributions[order],
        "pushes_toward": np.where(contributions[order] > 0, "AI", "Human"),
    })


# ---------------------------------------------------------------------------
# UI Pipeline execution sequence code layout
# ---------------------------------------------------------------------------
st.title("🕵️ AI vs Human Text Detector")
st.caption("Upload a document or paste text, choose an optimized model structure, and receive statistical classification breakdowns.")

tfidf = load_tfidf()
ling_scaler = load_ling_scaler()
sklearn_models = load_sklearn_models()
keras_models = load_keras_models()
tokenizer = load_embedding_assets()

all_models = {"sklearn": sklearn_models, "keras": keras_models}
available_model_names = list(sklearn_models.keys()) + list(keras_models.keys())

if not available_model_names:
    st.error("No trained models found in `models/`. Make sure your notebook outputs are pushed to your deployment directory.")
    st.stop()

with st.sidebar:
    st.header("Input Space")
    input_mode = st.radio("Provide text via:", ["Paste text", "Upload file (PDF/DOCX)"])
    text_input = ""
    if input_mode == "Paste text":
        text_input = st.text_area("Paste the text to analyze:", height=250,
                                  placeholder="Paste an essay, article, or any passage here...")
    else:
        uploaded = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
        if uploaded is not None:
            with st.spinner("Extracting text layers..."):
                text_input = extract_text(uploaded)
            st.success(f"Extracted {len(text_input.split())} words.")
            with st.expander("Preview extracted text"):
                st.write(text_input[:2000] + ("..." if len(text_input) > 2000 else ""))

    st.header("Model Parameters")
    model_choice = st.selectbox("Choose a classifier:", available_model_names)
    missing_dl = [n for n in ["FNN (Keras)", "LSTM", "CNN"] if n not in keras_models]
    if missing_dl:
        st.caption(f"Deep learning models currently inactive: {', '.join(missing_dl)}")
        
    if tokenizer is None and any(m in available_model_names for m in ["FNN (Keras)", "LSTM", "CNN"]):
        st.sidebar.error("⚠️ 'tokenizer.json' is missing from 'models/'. Deep Learning models will fail until it is uploaded.")

    run_btn = st.button("🔍 Run Prediction Framework", type="primary", use_container_width=True)

tab_predict, tab_compare, tab_report = st.tabs(["Prediction Dashboard", "Model Comparison Matrix", "Downloadable Report"])

if run_btn and text_input.strip():
    word_count = len(text_input.split())
    if word_count < 20:
        st.warning("This sequence string length is short — accuracy readings stabilize on sequences above 50+ words.")

    # Extract styling signals uniformly upfront to ensure metrics are always populated
    ling_feats = extract_linguistic_features(simple_clean_text(text_input))

    # ---- Single-model prediction ----
    with tab_predict:
        try:
            pred, proba = predict_with_model(model_choice, text_input, all_models, tfidf, ling_scaler, tokenizer)
        except Exception as e:
            st.error(f"Prediction logic execution fault encountered: {e}")
            pred, proba = None, None

        if pred is not None:
            label = "🤖 AI-Generated Text Detected" if pred == 1 else "🧑 Authentic Human-Written Text"
            confidence = proba if pred == 1 else 1 - proba
            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("Classifier Result Matrix", label)
                st.metric("Model Confidence Evaluation", f"{confidence*100:.1f}%")
                st.progress(float(confidence))
            with c2:
                st.write("**Extracted Document Linguistic Signal Values**")
                st.write(f"- Absolute Token Word Count: {word_count}")
                st.write(f"- Mean Sentence Evaluation Span: {ling_feats['avg_sentence_length']:.2f} words")
                st.write(f"- Vocabulary Structural Richness (TTR Ratio): {ling_feats['type_token_ratio']:.3f}")
                st.write(f"- Document Flesch Reading Score Profile: {ling_feats['flesch_reading_ease']:.1f}")
                st.write(f"- Syntactic Phrase Contraction Density: {ling_feats['contraction_ratio']:.4f}")

            st.subheader("Why this prediction? (Word-Level Attribute Contribution Analysis)")
            exp_df = explain_prediction(text_input, sklearn_models, tfidf)
            if exp_df is not None and len(exp_df):
                fig, ax = plt.subplots(figsize=(7, 4))
                colors = exp_df["contribution"].apply(lambda v: "#DD8452" if v > 0 else "#4C72B0")
                ax.barh(exp_df["term"][::-1], exp_df["contribution"][::-1], color=colors[::-1])
                ax.set_xlabel("Contribution Factor Index (← Human Style | AI Signature →)")
                st.pyplot(fig)
                st.caption("Explanation derived from the SVM's linear weights vector mapping across the document's local TF-IDF matrices.")
            else:
                st.info("Word-level explanation visualizations require the SVM classifier base weights vector to be active.")

    # ---- Side-by-side model comparison ----
    with tab_compare:
        st.subheader("Comparative Grid Model Evaluation Metrics Matrix")
        rows = []
        for name in available_model_names:
            try:
                p, pr = predict_with_model(name, text_input, all_models, tfidf, ling_scaler, tokenizer)
                conf = pr if p == 1 else 1 - pr
                rows.append({"Model Name": name, "Prediction Class": "AI Generated" if p == 1 else "Human Document",
                             "Confidence Index": f"{conf*100:.1f}%", "P(AI Probability Value)": round(float(pr), 4)})
            except Exception as e:
                rows.append({"Model Name": name, "Prediction Class": "Execution Exception Fault", "Confidence Index": "-", "P(AI Probability Value)": str(e)})
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        plot_df = comp_df[comp_df["P(AI Probability Value)"].apply(lambda x: isinstance(x, float))]
        if len(plot_df):
            ax.bar(plot_df["Model Name"], plot_df["P(AI Probability Value)"], color="#DD8452")
            ax.axhline(0.5, color="black", linestyle="--")
            ax.set_ylabel("P(AI Probability Output Space)")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=15)
            st.pyplot(fig)

    # ---- Downloadable report ----
    with tab_report:
        st.subheader("Export Project Verification Metric Logs")
        report_lines = [
            "GRADUATE RESEARCH PROJECT DELIVERABLE LOG REPORT",
            f"Execution Verification Timestamp (UTC): {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"Input Analysis Evaluation Token Metric Count: {word_count}",
            "",
            f"Active Evaluation Model Selection ID: {model_choice}",
            f"Calculated Inference Class Categorization: {'AI-Generated' if ('pred' in locals() and pred == 1) else 'Human-Written' if ('pred' in locals() and pred == 0) else 'Error State'}",
            f"Computed Confidence Reading Boundary Level: {confidence*100:.2f}%" if ('confidence' in locals() and pred is not None) else "N/A",
            "",
            "Complete Matrix Performance Mapping Log Data:",
        ]
        for r in rows:
            report_lines.append(f"  - {r['Model Name']}: {r['Prediction Class']} ({r['Confidence Index']})")
        report_lines += ["", "Extracted Stylistic Features Signature Value Metrics:"]
        for k, v in ling_feats.items():
            report_lines.append(f"  - Layer Attribute ID {k}: {v:.4f}")
        report_text = "\n".join(report_lines)
        st.text_area("Report text buffer snapshot preview data", report_text, height=300)
        st.download_button("⬇️ Download analysis report (.txt)", report_text,
                             file_name="ai_human_detection_report.txt")
else:
    with tab_predict:
        st.info("Input a document context string or attach a text file asset, then select 'Run Prediction Framework' to evaluate.")
