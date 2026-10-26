from pathlib import Path
from flask import Flask, jsonify, render_template, request
from model import HandwrittenDigitClassifier

# 1. Initialize the Flask web application and load our trained AI model
PROJECT_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
classifier = HandwrittenDigitClassifier(PROJECT_DIR)


# 2. Show the main webpage (templates/index.html)
@app.get("/")
def home():
    return render_template("index.html")


# 3. Receive the drawn image from JavaScript and return the predicted digit
@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    expected_digit = data.get("expected_digit", "")

    if not image_data.startswith("data:image/"):
        return jsonify({"error": "Draw a digit or upload a clear image first."}), 400

    try:
        result = classifier.predict(image_data, expected_digit)
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


# 4. Return basic information about the model
@app.get("/api/model-info")
def model_info():
    return jsonify({
        "model_type": "K-Nearest Neighbors (KNN)",
        "accuracy": classifier.accuracy,
        "dataset": "sklearn built-in handwritten digits (8x8)",
        "dataset_size": classifier.DATASET_SIZE,
        "classes": list(range(10)),
    })


if __name__ == "__main__":
    app.run(debug=True)