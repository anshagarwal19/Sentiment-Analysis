import string
import warnings
import pickle
import os

import streamlit as st

from pathlib import Path

# BASE_DIR = Path(__file__).parent

# st.write("Current working directory:", os.getcwd())
# st.write("BASE_DIR:", BASE_DIR)
# st.write("Files in BASE_DIR:", os.listdir(BASE_DIR))
# st.write("Model exists:", (BASE_DIR / "sentiment_model.pkl").exists())





warnings.filterwarnings("ignore")



def remove_punc(txt: str) -> str:
    return txt.translate(str.maketrans("", "", string.punctuation))


def remove_num(txt: str) -> str:
    return "".join(ch for ch in txt if not ch.isdigit())


def remove_emoji(txt: str) -> str:
    # Keeps only ASCII characters.
    return "".join(ch for ch in txt if ch.isascii())


def remove_stop_words(txt: str, stop_words: set[str]) -> str:
    words = txt.split()
    cleaned = [w for w in words if w not in stop_words]
    return " ".join(cleaned)


def preprocess(text: str, stop_words: set[str]) -> str:
    text = str(text).lower()
    text = remove_punc(text)
    text = remove_num(text)
    text = remove_emoji(text)
    text = remove_stop_words(text, stop_words)
    return text


BASE_DIR = Path(__file__).parent

@st.cache_resource(show_spinner=False)
def load_artifacts():
    pkl_path = BASE_DIR / "sentiment_model.pkl"

    if not pkl_path.exists():
        st.error(f"Missing model file: {pkl_path}")
        st.stop()

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    return (
        payload["model"],
        payload["tfidf_vectorizer"],
        payload["inv_emotion_numbers"],
        payload["stop_words"],
    )


def predict_emotion(text: str) -> str:
    model, vectorizer, inv_emotion_numbers, stop_words = load_artifacts()
    processed = preprocess(text, stop_words)
    vec = vectorizer.transform([processed])
    pred_num = int(model.predict(vec)[0])
    return inv_emotion_numbers[pred_num]



st.set_page_config(page_title="Sentiment Emotion Predictor", layout="centered")

st.title("Emotion Predictor (TF-IDF + Logistic Regression)")
st.caption("Predicts emotions using ML")

user_text = st.text_area("Enter your text:", height=160, placeholder="Type something like: 'I feel really angry and rejected' ...")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Predict", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.rerun()

if run_btn and user_text.strip():
    with st.spinner("Predicting..."):
        emotion = predict_emotion(user_text)
    st.success(f"Predicted emotion: **{emotion}**")

st.divider()


