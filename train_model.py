# train_model.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
def load_dataset(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":::")
            if len(parts) >= 4:
                movie_id, title, genre, plot = parts[0], parts[1], parts[2], ":::".join(parts[3:])
                rows.append((movie_id, title, genre.lower(), plot))
    return pd.DataFrame(rows, columns=["id", "title", "genre", "plot"])

# ------------------------------
# Step 2: Preprocess Text
# ------------------------------
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
    return text

# ------------------------------
# Step 3: Train Models
# ------------------------------
def train_models(X_train, y_train, X_val, y_val):
    results = {}
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=300),
        "SVM": LinearSVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results[name] = (acc, model)
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_val, preds))

    return results

# ------------------------------
# Step 4: Main Function
# ------------------------------
def main():
    print("Loading dataset...")
    df = load_dataset("data/train_data.txt")
    print(f"Dataset loaded: {df.shape[0]} rows")

    # Preprocess
    df["clean_plot"] = df["plot"].apply(preprocess_text)

    # Split train/test
    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_plot"], df["genre"], test_size=0.2, random_state=42
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Train models
    results = train_models(X_train_tfidf, y_train, X_val_tfidf, y_val)

    # Pick best model
    best_model_name = max(results, key=lambda k: results[k][0])
    best_acc, best_model = results[best_model_name]
    print(f"\n✅ Best Model: {best_model_name} with accuracy {best_acc:.4f}")

    # Save vectorizer + model
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(best_model, "movie_genre_model.pkl")
    print("✅ Model and vectorizer saved!")

if __name__ == "__main__":
    main()
