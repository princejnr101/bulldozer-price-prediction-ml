from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.file_utils import find_kaggle_csv
from src.preprocess import add_dateparts

MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(exist_ok=True)

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(0, y_pred)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred_clip = np.maximum(0, y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmsle": float(np.sqrt(mean_squared_log_error(y_true, y_pred_clip))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def main():
    train_path = find_kaggle_csv("TrainAndValid")
    df = pd.read_csv(train_path, low_memory=False, parse_dates=["saledate"])
    df = df.sort_values("saledate").reset_index(drop=True)

    split_date = pd.to_datetime("2012-01-01")
    train_df = df[df["saledate"] < split_date].copy()
    valid_df = df[df["saledate"] >= split_date].copy()

    train_df = add_dateparts(train_df, "saledate", drop=True)
    valid_df = add_dateparts(valid_df, "saledate", drop=True)

    target = "SalePrice"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].values
    X_valid = valid_df.drop(columns=[target])
    y_valid = valid_df[target].values

    for c in X_train.columns:
        if X_train[c].dtype == "object":
            X_train[c] = X_train[c].astype("category")
    for c in X_valid.columns:
        if X_valid[c].dtype == "object":
            X_valid[c] = X_valid[c].astype("category")

    dt_cols = list(X_train.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns)
    if dt_cols:
        X_train = X_train.drop(columns=dt_cols)
        X_valid = X_valid.drop(columns=dt_cols, errors="ignore")

    X_valid = X_valid.reindex(columns=X_train.columns)

    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) in ("category", "object")]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    pipe = Pipeline([("preprocess", pre), ("model", model)])

    pipe.fit(X_train, y_train)
    base_pred = pipe.predict(X_valid)
    baseline = evaluate(y_valid, base_pred)

    scorer = make_scorer(lambda yt, yp: -rmsle(yt, yp), greater_is_better=True)

    param_dist = {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [None, 10, 20, 30],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring=scorer,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    tuned_pred = best.predict(X_valid)
    tuned = evaluate(y_valid, tuned_pred)

    joblib.dump(best, MODELS_DIR/"model.joblib")

    out = {
        "baseline": baseline,
        "tuned": tuned,
        "best_params": search.best_params_,
        "train_path_used": str(train_path),
        "split_date": "2012-01-01",
    }
    (MODELS_DIR/"metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    rf = best.named_steps["model"]
    importances = rf.feature_importances_
    feature_names = num_cols + cat_cols
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(REPORTS_DIR/"feature_importance.csv", index=False)

    top = fi.head(20)[::-1]
    plt.figure(figsize=(8, 10))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 20 Feature Importances (RandomForest)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR/"feature_importance_top20.png", dpi=200)
    plt.close()

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
