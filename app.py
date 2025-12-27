from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocess import add_dateparts

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="Bulldozer SalePrice Predictor", version="1.1.0")

class PredictRequest(BaseModel):
    saledate: str = Field(..., description="ISO date string, e.g. 2012-11-19")
    features: Dict[str, Any] = Field(default_factory=dict, description="Other feature columns")

class PredictResponse(BaseModel):
    prediction: float

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Train first to create models/model.joblib")

    df = pd.DataFrame([{**{"saledate": req.saledate}, **req.features}])
    df = add_dateparts(df, "saledate", drop=True)

    dt_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns)
    if dt_cols:
        df = df.drop(columns=dt_cols)

    pred = float(model.predict(df)[0])
    return {"prediction": max(0.0, pred)}
