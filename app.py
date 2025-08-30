from flask import Flask, request, jsonify, render_template
import joblib

# --------------------------
# Load Model + Vectorizer
# --------------------------
model = joblib.load("movie_genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --------------------------
# Initialize Flask App
# --------------------------
app = Flask(__name__)

# --------------------------
# API Endpoint: Predict
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    plot = data.get("plot", "")

    # Preprocess input
    plot_clean = plot.lower()
    plot_clean = plot_clean.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))

    X = vectorizer.transform([plot_clean])
    prediction = model.predict(X)[0]

    return jsonify({"genre": prediction})

# --------------------------
# Frontend Page
# --------------------------
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
