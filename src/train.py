import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from src.preprocessing import clean_text


def train_model(data_path: str):
    df = pd.read_csv(data_path)

    df = df[df["label"].isin(["negative", "neutral", "positive"])]

    df["text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        )
    )

    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")


if __name__ == "__main__":
    train_model("data/rusentitweet_full.csv")
