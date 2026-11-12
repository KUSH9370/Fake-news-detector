
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", str(text)).lower()
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def main():
    df = pd.read_csv("News.csv")
    if not {"text", "class"}.issubset(df.columns):
        raise ValueError("News.csv needs columns named 'text' and 'class'.")
    df = df.dropna(subset=["text", "class"]).copy()
    df["clean_text"] = df["text"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["class"], test_size=0.25,
        random_state=42, stratify=df["class"]
    )
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Matches the last classifier used in the uploaded notebook.
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, "news_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    joblib.dump({
        "model_name": "Decision Tree",
        "features": "TF-IDF",
        "test_accuracy": float(acc),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }, "model_metadata.joblib")

    print(f"Held-out test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Saved model and vectorizer artifacts.")
    print("Check dataset labels: app assumes 1=Real News and 0=Fake News.")

if __name__ == "__main__":
    main()
