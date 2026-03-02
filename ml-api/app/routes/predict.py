"""
Prediction route.
Purpose: POST /predict endpoint that delegates to the predictor service.
Modify: Add validation, preprocessing, or logging here.
"""
from fastapi import APIRouter, HTTPException

from app.models.schema import PredictionInput, PredictionOutput
from app.services.predictor import predictor

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Accept input features and return prediction."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model and place model.pkl in ml-api/model/",
        )
    result = predictor.predict(input_data.features)
    return PredictionOutput(
        prediction=result["prediction"],
        confidence=result["confidence"],
        label=result.get("label", str(result["prediction"])),
    )
