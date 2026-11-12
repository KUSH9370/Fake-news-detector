
import os
import re
import joblib
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Verity | News Detection", page_icon="📰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0b1020;color:#edf2ff}
[data-testid="stSidebar"]{background:#10182b;border-right:1px solid #25314a}
[data-testid="stHeader"]{background:rgba(11,16,32,.85)}
.block-container{padding-top:2rem;max-width:1250px}
.hero{padding:28px 30px;border:1px solid #273653;border-radius:22px;background:linear-gradient(120deg,#172541,#142039 55%,#192d3b);margin-bottom:22px}
.eyebrow{color:#7dd3fc;font-size:12px;letter-spacing:2px;font-weight:700;text-transform:uppercase}
.hero h1{font-family:Manrope,sans-serif;font-size:36px;margin:8px 0 6px;color:#f4f7ff}
.hero p,.muted{color:#aab8d2}
.panel,.metric{background:#121c30;border:1px solid #273653;border-radius:18px;padding:20px 22px;margin-bottom:16px}
.metric{min-height:95px}
.metric-label{color:#91a2c0;font-size:12px;text-transform:uppercase;letter-spacing:1px}
.metric-value{color:#f2f6ff;font-family:Manrope,sans-serif;font-size:25px;font-weight:800;margin-top:7px}
.result-real{background:#102e2a;border:1px solid #1e7664;border-radius:18px;padding:22px}
.result-fake{background:#351c29;border:1px solid #9c3c5a;border-radius:18px;padding:22px}
.result-title{font-family:Manrope,sans-serif;font-size:27px;font-weight:800;margin:5px 0}
div.stButton>button{background:linear-gradient(90deg,#38bdf8,#6366f1);color:white;border:0;border-radius:10px;font-weight:700;padding:.65rem 1.2rem}
textarea{background:#0d1628!important;color:#edf2ff!important}
hr{border-color:#263550}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "news_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
META_PATH = "model_metadata.joblib"

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", str(text)).lower()
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None, {}
    m = joblib.load(MODEL_PATH)
    v = joblib.load(VECTORIZER_PATH)
    meta = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
    return m, v, meta

model, vectorizer, metadata = load_artifacts()

if "checks" not in st.session_state:
    st.session_state.checks = []

with st.sidebar:
    st.markdown("## 📰 VERITY")
    st.caption("NEWS INTELLIGENCE")
    page = st.radio("WORKSPACE", ["News detector", "Model performance", "About"])
    st.markdown("---")
    if model is None:
        st.warning("Model files not found")
        st.caption("Run `python train_model.py` first.")
    else:
        st.success("Model loaded")
        st.caption(f"Model: {metadata.get('model_name', 'Decision Tree')}")

st.markdown("""
<div class="hero">
 <div class="eyebrow">AI-powered media analysis</div>
 <h1>Know what you're reading.</h1>
 <p>Paste your own news article, get an AI model prediction, and optionally compare it with the known label.</p>
</div>
""", unsafe_allow_html=True)

if page == "News detector":
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Analyze your news")
        news = st.text_area("Paste news article", height=260,
                            placeholder="Paste the complete news article or headline here...")
        st.caption(f"Characters: {len(news)} · Words: {len(news.split())}")
        run = st.button("✦ Analyze this news", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("How to use")
        st.markdown("""
        1. Paste your news text.
        2. Click **Analyze this news**.
        3. Review the predicted class and model confidence.
        4. If you know the true label, select it to measure whether this prediction was correct.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("A single article cannot provide an accuracy score by itself unless you know its correct label. Accuracy is measured across multiple labeled examples.")

    if run:
        if model is None:
            st.error("Model files are missing. Put News.csv in this folder and run `python train_model.py`.")
        elif len(news.strip().split()) < 5:
            st.warning("Please enter at least 5 words for a more meaningful prediction.")
        else:
            X = vectorizer.transform([clean_text(news)])
            pred = model.predict(X)[0]
            # Notebook's prediction function assumes class 1 = Real, class 0 = Fake.
            real_label = str(pred).strip().lower() in ("1", "1.0", "true")
            label = "Real News" if real_label else "Potentially Fake News"
            probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
            confidence = float(max(probs) * 100) if probs is not None else None
            st.session_state.latest = {
                "text": news, "prediction": label, "pred_value": str(pred),
                "confidence": confidence
            }

    if "latest" in st.session_state:
        result = st.session_state.latest
        if result["prediction"] == "Real News":
            st.markdown(f'<div class="result-real"><div class="eyebrow" style="color:#6ee7b7">MODEL PREDICTION</div><div class="result-title">✓ {result["prediction"]}</div><div>The model assigned this text to its Real News class.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-fake"><div class="eyebrow" style="color:#fda4af">MODEL PREDICTION</div><div class="result-title">! {result["prediction"]}</div><div>The model assigned this text to its Fake News class.</div></div>', unsafe_allow_html=True)
        if result["confidence"] is not None:
            st.metric("Model confidence (not factual certainty)", f'{result["confidence"]:.1f}%')
            st.progress(result["confidence"] / 100)

        st.markdown("#### Do you know the actual label?")
        actual = st.radio("Choose the verified label to evaluate this prediction",
                          ["Not sure", "Real News", "Fake News"], horizontal=True,
                          key="actual_label")
        if st.button("Save this evaluation"):
            if actual == "Not sure":
                st.warning("Choose a known label to include this item in the accuracy calculation.")
            else:
                correct = (actual == result["prediction"]) or (
                    actual == "Fake News" and result["prediction"] == "Potentially Fake News"
                )
                st.session_state.checks.append({
                    "Prediction": result["prediction"],
                    "Actual label": actual,
                    "Correct": "Yes" if correct else "No",
                    "Confidence": result["confidence"]
                })
                st.success("Evaluation saved to this session.")
        st.warning("This is a text-pattern classifier, not a fact-checking service. Verify important claims using reliable independent sources.")

elif page == "Model performance":
    st.subheader("Model performance")
    if metadata:
        a, b, c = st.columns(3)
        with a:
            st.markdown(f'<div class="metric"><div class="metric-label">Held-out test accuracy</div><div class="metric-value">{metadata.get("test_accuracy", 0):.1%}</div></div>', unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="metric"><div class="metric-label">Training samples</div><div class="metric-value">{metadata.get("training_rows", "—")}</div></div>', unsafe_allow_html=True)
        with c:
            st.markdown(f'<div class="metric"><div class="metric-label">Test samples</div><div class="metric-value">{metadata.get("test_rows", "—")}</div></div>', unsafe_allow_html=True)
        st.caption("These metrics come from the held-out dataset during training. They are not the confidence score for an individual article.")
    else:
        st.info("No training metrics found. Run `python train_model.py` to train and evaluate the model.")
    st.markdown("### Your manually evaluated articles")
    checks = st.session_state.checks
    if checks:
        df = pd.DataFrame(checks)
        correct_n = sum(df["Correct"] == "Yes")
        total = len(df)
        st.metric("Accuracy on labeled articles in this session", f"{correct_n/total:.1%}", f"{correct_n} correct out of {total}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download evaluation results (CSV)", csv, "news_evaluations.csv", "text/csv")
    else:
        st.caption("No manually labeled articles yet. Analyze news and save an evaluation when you know its actual label.")

elif page == "About":
    st.subheader("How the classifier works")
    st.markdown("""
    **Your notebook's pipeline:**
    1. Clean text: lowercase, remove punctuation and English stopwords.
    2. TF-IDF: convert text into numerical word-weight features.
    3. Decision Tree: classify the feature vector into a dataset label.

    **Important label mapping:** the notebook's prediction function assumes `1 = Real News` and `0 = Fake News`. Check that this matches your `News.csv`.

    **What the numbers mean:**
    - **Prediction:** the class selected by the classifier.
    - **Confidence:** the model's estimated class probability, not the probability that the article is factually true.
    - **Accuracy:** the fraction of labeled test examples classified correctly.
    """)
    st.warning("Fake/real news detection can be wrong, especially for satire, opinion, breaking news, and topics unlike the training dataset.")
st.markdown("---")
st.markdown('<p class="muted" style="text-align:center">VERITY • News classification demo • Verify important claims independently.</p>', unsafe_allow_html=True)
