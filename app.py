"""FastAPI server exposing the trained model.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
Docs:     http://localhost:8000/docs
"""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_pipeline import load_model, predict_model

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")

app = FastAPI(title="Breast Cancer Prediction API")


class PredictionRequest(BaseModel):
    """Input: a list of samples, each a {feature_name: value} mapping."""

    samples: list[dict[str, float]]


@app.get("/")
def health():
    """Health check."""
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Return the predicted class and malignant probability per sample."""
    try:
        model, scaler = load_model(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    features = pd.DataFrame(request.samples)
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in features.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    labels, probas = predict_model(model, scaler, features[expected])
    return {
        "predictions": [
            {
                "diagnosis": "malignant" if y == 1 else "benign",
                "label": y,
                "probability_malignant": round(p, 4),
            }
            for y, p in zip(labels, probas)
        ]
    }
