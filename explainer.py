
import re
import numpy as np
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from test_model import model, tokenizer

MAX_LEN = 150

def normalize_arabic_text_v2(text):
    text = str(text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def arabic_splitter(text):
    return re.findall(r'\S+', text)
explainer = LimeTextExplainer(
    class_names=["Not Scam", "Scam"],
    split_expression=arabic_splitter
)

def predict_with_lstm(texts):
    normalized = [
        normalize_arabic_text_v2(text)
        for text in texts
    ]

    sequences = tokenizer.texts_to_sequences(normalized)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    scam_probabilities = model.predict(
        padded, verbose=0
    ).flatten()

    return np.column_stack([
        1 - scam_probabilities,
        scam_probabilities
    ])

def explain_conversation(text):
    probabilities = predict_with_lstm([text])[0]
    predicted_label = int(probabilities[1] >= 0.5)

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_with_lstm,
        labels=(predicted_label,),
        num_features=10,
        num_samples=1000
    )

    return {
        "scam_probability": round(float(probabilities[1] * 100), 2),
        "not_scam_probability": round(float(probabilities[0] * 100), 2),
        "result": "احتيال محتمل" if predicted_label else "محادثة سليمة محتملة",
        "important_words": explanation.as_list(label=predicted_label)
    }
