import string
import warnings
import pickle
import os

import streamlit as st

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


@st.cache_resource(show_spinner=False)
def load_artifacts(pkl_path: str = "sentiment_model.pkl"):
    if not os.path.exists(pkl_path):
        st.error(
            "Missing `sentiment_model.pkl`. Run `python train_and_dump.py --train-path train.txt --out-path sentiment_model.pkl` once."
        )
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
st.caption("Predicts emotions")

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


