import joblib

# --------------------------
# Load Model + Vectorizer
# --------------------------
model = joblib.load("movie_genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --------------------------
# Predict Function
# --------------------------
def predict_genre(plot):
    """
    Takes a movie plot string and returns predicted genre.
    """
    plot_clean = plot.lower()
    plot_clean = plot_clean.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    
    X = vectorizer.transform([plot_clean])
    prediction = model.predict(X)[0]
    return prediction

# --------------------------
# Example Run
# --------------------------
if __name__ == "__main__":
    sample_plot = "A young boy discovers he has magical powers and attends a special school for wizards."
    print(f"Plot: {sample_plot}")
    print(f"Predicted Genre: {predict_genre(sample_plot)}")
