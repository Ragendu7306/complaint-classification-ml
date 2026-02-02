from flask import Flask, request, jsonify
import joblib
import re

# Load model & vectorizer
model = joblib.load("ml_models/complaint_model.pkl")
vectorizer = joblib.load("ml_models/tfidf_vectorizer.pkl")

# Create Flask app
app = Flask(__name__)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# Home route
@app.route("/")
def home():
    return "✅ Complaint Classification API is running"

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    cleaned = clean_text(data["text"])
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return jsonify({
        "complaint_text": data["text"],
        "predicted_product": prediction[0]
    })

# Run server
if __name__ == "__main__":
    app.run(debug=True)
