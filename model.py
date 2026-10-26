import base64
from io import BytesIO
import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class HandwrittenDigitClassifier:
    DATASET_SIZE = 1797  # Total images in sklearn's built-in 8x8 digit dataset

    def __init__(self, project_dir):
        self.model_path = project_dir / "artifacts" / "digit_model_knn.joblib"
        self.model, self.accuracy = self._load_or_train()

    def _load_or_train(self):
        # 1. If the model was already trained and saved earlier, load it directly
        if self.model_path.exists():
            saved = joblib.load(self.model_path)
            print(f"  Loaded KNN model with {saved['accuracy']}% accuracy.")
            return saved["model"], saved["accuracy"]

        # 2. Load the 1,797 built-in handwritten digit images (each is 8x8 pixels)
        digits = load_digits()
        features = np.array([self._extract_features(img / 16 * 255) for img in digits.images])

        # 3. Split data: 80% for training, 20% for testing accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            features, digits.target, test_size=0.2, random_state=42
        )

        # 4. Train K-Nearest Neighbors (compares a drawing to the 3 closest examples)
        eval_model = KNeighborsClassifier(n_neighbors=3, weights="distance")
        eval_model.fit(X_train, y_train)
        accuracy = round(float(accuracy_score(y_test, eval_model.predict(X_test))) * 100, 2)
        print(f"  Trained KNN Model | Accuracy: {accuracy}% | Train: {len(X_train)} | Test: {len(X_test)}")

        # 5. Train final model on 100% of the data and save it to disk
        model = KNeighborsClassifier(n_neighbors=3, weights="distance")
        model.fit(features, digits.target)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "accuracy": accuracy}, self.model_path)
        return model, accuracy

    def _extract_features(self, pixels):
        # 1. Find where the bright ink pixels are located (ignore blank background)
        threshold = max(30, pixels.max() * 0.16)
        rows, cols = np.where(pixels > threshold)
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("No clear digit was found. Draw one large digit in the centre of the box.")

        # 2. Crop tightly around the drawn digit
        digit = pixels[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

        # 3. Resize digit to fit inside a 20x20 box while keeping its aspect ratio
        height, width = digit.shape
        scale = 20 / max(height, width)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = Image.fromarray(digit.astype(np.uint8)).resize(new_size, Image.Resampling.LANCZOS)

        # 4. Paste into the center of a 32x32 black square, then shrink to 8x8 (64 numbers)
        board = Image.new("L", (32, 32), 0)
        board.paste(resized, ((32 - resized.width) // 2, (32 - resized.height) // 2))
        small = board.resize((8, 8), Image.Resampling.LANCZOS)

        # 5. Convert brightness from 0-255 scale to sklearn's 0-16 scale
        return np.asarray(small, dtype=np.float32).flatten() / 255 * 16

    def predict(self, image_data, expected_digit):
        # 1. Decode the base64 image from the browser and convert to grayscale ("L")
        try:
            encoded = image_data.split(",", 1)[1]
            raw_image = Image.open(BytesIO(base64.b64decode(encoded))).convert("L")
        except Exception:
            raise ValueError("The image could not be read. Please use a PNG or JPG image.")

        pixels = np.asarray(raw_image, dtype=np.float32)

        # 2. Invert colors if background is white (make digit bright on dark background)
        if pixels.mean() > 127:
            pixels = 255 - pixels
        pixels = np.clip((pixels - 25) * 1.2, 0, 255)

        # 3. Turn the image into 64 numbers and get probabilities for digits 0-9
        features = self._extract_features(pixels)
        probabilities = self.model.predict_proba([features])[0]
        ranked = np.argsort(probabilities)[::-1]

        predicted_digit = int(self.model.classes_[ranked[0]])
        confidence = round(float(probabilities[ranked[0]]) * 100)

        # 4. Check if the user entered an optional expected digit (0-9)
        expected_str = "" if expected_digit is None else str(expected_digit).strip()
        expected = int(expected_str) if expected_str.isdigit() and int(expected_str) in range(10) else None
        if expected_str != "" and expected is None:
            raise ValueError("Expected digit must be a whole number from 0 to 9.")

        return {
            "prediction": predicted_digit,
            "confidence": confidence,
            "expected_digit": expected,
            "is_match": None if expected is None else (predicted_digit == expected),
            "accuracy": self.accuracy,
            "dataset_size": self.DATASET_SIZE,
            "top_predictions": [
                {"digit": int(self.model.classes_[i]), "confidence": round(float(probabilities[i]) * 100)}
                for i in ranked[:3]
            ],
        }