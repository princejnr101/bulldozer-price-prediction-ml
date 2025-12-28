from __future__ import annotations

import os
import re
import numpy as np

import urllib.request
import http.cookiejar
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocess import add_dateparts

MODEL_PATH = Path("models/model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Bulldozer SalePrice Predictor", version="1.3.0")


class PredictRequest(BaseModel):
    saledate: str = Field(..., description="ISO date string, e.g. 2012-11-19")
    features: Dict[str, Any] = Field(default_factory=dict, description="Other feature columns")


class PredictResponse(BaseModel):
    prediction: float


def _looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(256).lstrip()
        return head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or head.startswith(b"<")
    except Exception:
        return False


def _download_stream(opener, url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=120) as resp:
        dest_tmp = dest.with_suffix(dest.suffix + ".tmp")
        with dest_tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                out.write(chunk)
        dest_tmp.replace(dest)


def _extract_drive_file_id(url: str) -> str | None:
    # Handles:
    # https://drive.google.com/file/d/<ID>/view?...
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        return m.group(1)
    # https://drive.google.com/uc?export=download&id=<ID>
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        return m.group(1)
    return None


def download_from_google_drive(shared_or_uc_url: str, dest: Path) -> None:
    """
    Robust Google Drive downloader that handles the large-file confirmation page.
    """
    file_id = _extract_drive_file_id(shared_or_uc_url)
    if not file_id:
        raise RuntimeError("Could not extract Google Drive file id from MODEL_URL")

    base = f"https://drive.google.com/uc?export=download&id={file_id}"

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    # 1) First request
    req1 = urllib.request.Request(base, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req1, timeout=120) as resp1:
        content_type = resp1.headers.get("Content-Type", "")
        content_disp = resp1.headers.get("Content-Disposition", "")
        body = resp1.read(200000)  # read enough to detect confirm token if HTML

    # If we already got a file, just stream it (redo streaming because we consumed body)
    if "attachment" in content_disp.lower():
        _download_stream(opener, base, dest)
        return

    # Otherwise it is likely HTML confirmation page; find confirm token
    text = body.decode("utf-8", errors="ignore")
    token_match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
    if not token_match:
        # Sometimes Drive blocks or rate-limits
        raise RuntimeError(
            f"Google Drive did not provide a confirm token. Content-Type={content_type}. "
            "Your file may be too large, not shared publicly, or rate-limited."
        )

    token = token_match.group(1)
    url2 = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
    _download_stream(opener, url2, dest)


def ensure_model_file() -> None:
    if MODEL_PATH.exists():
        return

    url = os.getenv("MODEL_URL")
    if not url:
        raise RuntimeError("MODEL_URL env var is not set and models/model.joblib is missing.")

    print(f"[startup] Downloading model from MODEL_URL -> {MODEL_PATH} ...")

    # Google Drive-safe handling
    if "drive.google.com" in url:
        download_from_google_drive(url, MODEL_PATH)
    else:
        # generic direct download
        urllib.request.urlretrieve(url, MODEL_PATH)

    # Validate download
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"[startup] Model downloaded. Size: {size_mb:.2f} MB")

    if size_mb < 5:
        raise RuntimeError(f"Downloaded file is suspiciously small ({size_mb:.2f} MB). Bad link/HTML page likely.")

    if _looks_like_html(MODEL_PATH):
        raise RuntimeError("Downloaded file looks like HTML (Google Drive confirmation page). Not a joblib model.")


model = None
load_error = None

try:
    ensure_model_file()
    model = joblib.load(MODEL_PATH)
    print("[startup] Model loaded successfully.")
except Exception as e:
    # Make error useful (type + message)
    load_error = f"{type(e).__name__}: {e}"
    print("[startup] Model load failed:", load_error)
    model = None


@app.get("/")
def root():
    return {"message": "Bulldozer SalePrice Predictor API", "health": "/health", "predict": "/predict"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path_exists": MODEL_PATH.exists(),
        "model_size_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None,
        "load_error": load_error,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=LOAD_ERROR or "Model not loaded")

    df = pd.DataFrame([{**{"saledate": req.saledate}, **req.features}])
    df = add_dateparts(df, "saledate", drop=True)

    # Drop any datetime columns that survived
    dt_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns)
    if dt_cols:
        df = df.drop(columns=dt_cols)

    def _try_predict(_df: pd.DataFrame) -> float:
        pred_val = float(model.predict(_df)[0])
        return max(0.0, pred_val)

    try:
        return {"prediction": _try_predict(df)}

    except ValueError as e:
        msg = str(e)

        # ✅ Handle: "columns are missing: {'A', 'B', ...}"
        if "columns are missing:" in msg:
            # Extract the {...} part
            m = re.search(r"columns are missing:\s*\{(.+)\}", msg)
            if not m:
                raise HTTPException(status_code=500, detail=f"ValueError: {msg}")

            inside = m.group(1)
            # Split by comma, strip quotes/spaces
            missing_cols = []
            for token in inside.split(","):
                col = token.strip().strip("'").strip('"')
                if col:
                    missing_cols.append(col)

            # Add missing as NaN so your imputers/encoders can handle them
            for c in missing_cols:
                if c not in df.columns:
                    df[c] = np.nan

            # Retry once
            return {"prediction": _try_predict(df)}

        # Otherwise: return real error
        raise HTTPException(status_code=500, detail=f"ValueError: {msg}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

