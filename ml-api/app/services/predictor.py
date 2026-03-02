"""
Predictor service - loads model and runs inference.
Purpose: Centralized model loading and prediction logic.
Modify: Change model path, add preprocessing, or support multiple models.

To retrain and replace the model:
1. Run: python scripts/train.py
2. Ensure ml-api/model/model.pkl exists
3. Restart the API
"""
import os
from pathlib import Path

import joblib


class PredictorService:
    def __init__(self):
        self._model = None
        self._model_path = Path(__file__).resolve().parent.parent.parent / "model" / "model.pkl"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """Load model.pkl from model/ directory. Fails gracefully if missing."""
        if not self._model_path.exists():
            return
        try:
            self._model = joblib.load(self._model_path)
        except Exception:
            self._model = None

    def predict(self, input_data: dict) -> dict:
        """Run inference. Returns {prediction, confidence, label?}."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        # Build feature vector from dict. Adapt to your model's expected input format.
        # Example: model expects [f1, f2, f3] in order
        feature_keys = sorted(input_data.keys())
        X = [[input_data[k] for k in feature_keys]]

        pred = self._model.predict(X)[0]

        # Try to get probabilities for confidence (classification models)
        try:
            probs = self._model.predict_proba(X)[0]
            confidence = float(max(probs))
        except AttributeError:
            confidence = 0.95  # Fallback for regression

        return {
            "prediction": int(pred) if hasattr(pred, "item") else float(pred),
            "confidence": confidence,
            "label": str(pred),
        }


# Singleton instance for use in routes
predictor = PredictorService()
