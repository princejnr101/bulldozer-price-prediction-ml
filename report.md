# Bluebook for Bulldozers — Report (RMSLE)

## Problem
Predict auction **SalePrice** for heavy equipment (regression).

## Data
Kaggle Bluebook for Bulldozers.

## Metric
Primary: **RMSLE** (matches Kaggle; relative error on skewed price distribution).

## Split
Time-based split with cutoff **2012-01-01** to avoid leakage.

## Preprocessing
- Parse `saledate`, sort chronologically
- Extract date parts and **drop raw datetime**
- object → category
- Impute numeric median; categorical most-frequent
- Ordinal encode categoricals (unknown → -1)

## Model
RandomForest baseline + RandomizedSearchCV tuning.

## Results / Feature Importance
Populate from `models/metrics.json` and `reports/feature_importance.csv` after training.
