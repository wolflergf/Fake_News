import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data["text"]
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)

    # Convert prediction to a native Python int
    prediction_label = int(prediction[0])

    return jsonify({"prediction": prediction_label})


if __name__ == "__main__":
    app.run(debug=True)
