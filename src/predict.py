import joblib
import numpy as np
from src.preprocessing import clean_text


class SentimentModel:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.labels = self.model.classes_

    def predict(self, text: str):
        text = clean_text(text)
        vec = self.vectorizer.transform([text])

        probs = self.model.predict_proba(vec)[0]
        best_idx = np.argmax(probs)

        return {
            "label": self.labels[best_idx],
            "confidence": float(probs[best_idx]),
            "probs": dict(zip(self.labels, probs))
        }
