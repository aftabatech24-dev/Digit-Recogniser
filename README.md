# Digit Recogniser

A small Flask web application that recognises handwritten digits from `0` to `9`. Draw a digit in the browser or upload a clear image, and a K-Nearest Neighbors (KNN) model returns its prediction, confidence, and alternative matches.

## Features

- Draw a digit directly on the canvas or upload a PNG, JPEG, or WebP image.
- Optional expected-digit check for quick testing.
- Shows prediction confidence and the top three likely digits.
- Uses scikit-learn's built-in handwritten-digits dataset.
- Trains and saves the model automatically the first time it runs.

## Project structure

```text
Digit-Recogniser/
├── app.py                 # Flask routes and API endpoints
├── model.py               # Image processing and KNN classifier
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Flask HTML template
└── static/
    ├── app.js             # Canvas and browser interactions
    └── styles.css         # Page styling
```

## Run locally

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the app.

   ```powershell
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in your browser.

The first launch trains the model and stores it locally in `artifacts/`. That generated file is intentionally not committed to the repository.

## Tech stack

- Python and Flask
- scikit-learn
- NumPy, Pillow, and joblib
- HTML, CSS, and JavaScript
