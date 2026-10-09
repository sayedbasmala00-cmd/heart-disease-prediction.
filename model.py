from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"


def load_model_bundle():
    """Load the saved model and its feature information."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. "
            "Place heart_disease_model.pkl inside the project folder."
        )

    return joblib.load(MODEL_PATH)


def predict_heart_disease(model, input_df):
    """Return the predicted class and probability if available."""
    prediction = int(model.predict(input_df)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        probability = float(probabilities[1])

    return prediction, probability