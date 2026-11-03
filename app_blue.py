
from flask import Flask, jsonify, request
import joblib, numpy as np

app = Flask(__name__)
model = joblib.load("model_blue.pkl")

# Endpoint 1 (Endpoint Tambahan/Lain): Health Check Status Server
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "version": "Blue", "model": "Wine Classifier (Logistic Regression)"}), 200

# Endpoint 2: Predict (13 Fitur Wine Dataset)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("data", [])
    pred = model.predict([np.array(data)]).tolist()
    return jsonify({"version": "Blue", "prediction": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
