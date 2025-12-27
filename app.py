from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocess import add_dateparts

MODEL_PATH = Path("models/model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Bulldozer SalePrice Predictor", version="1.2.0")


class PredictRequest(BaseModel):
    saledate: str = Field(..., description="ISO date string, e.g. 2012-11-19")
    features: Dict[str, Any] = Field(
        default_factory=dict, description="Other feature columns"
    )


class PredictResponse(BaseModel):
    prediction: float


def ensure_model_file() -> None:
    """
    Ensure the model file exists.
    If missing (common on Render), download it from MODEL_URL.
    """
    if MODEL_PATH.exists():
        return

    url = os.getenv("MODEL_URL")
    if not url:
        raise RuntimeError(
            "Model file is missing and MODEL_URL env var is not set. "
            "Set MODEL_URL to a direct-download link for models/model.joblib."
        )

    print(f"[startup] Downloading model from MODEL_URL -> {MODEL_PATH} ...")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[startup] Model downloaded successfully.")


model = None
load_error = None
try:
    ensure_model_file()
    model = joblib.load(MODEL_PATH)
except Exception as e:
    load_error = str(e)
    print("[startup] Model load failed:", e)
    model = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path_exists": MODEL_PATH.exists(),
        "load_error": load_error,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not available. load_error={load_error}",
        )

    # Build 1-row dataframe from request
    df = pd.DataFrame([{**{"saledate": req.saledate}, **req.features}])

    # Apply the exact same feature engineering as training
    df = add_dateparts(df, "saledate", drop=True)

    # Defensive: drop any remaining datetime cols (prevents dtype errors)
    dt_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns)
    if dt_cols:
        df = df.drop(columns=dt_cols)

    pred = float(model.predict(df)[0])
    return {"prediction": max(0.0, pred)}
