# Bluebook for Bulldozers — End-to-End Regression (RMSLE)

This project is designed to **avoid common notebook errors**:
- file path/name issues
- datetime dtype promotion errors
- stale column lists causing KeyError on dropped columns

## Put data in either location
**Option A (recommended):**
- `data/TrainAndValid.csv`
- `data/Test.csv`

**Option B:**
- `TrainAndValid.csv` and `Test.csv` beside the notebook/script.

The loader also handles common variants like `TrainAndValid (1).csv`.

## Train + Tune
```bash
pip install -r requirements.txt
python train.py
```

Outputs:
- `models/model.joblib`
- `models/metrics.json`
- `reports/feature_importance.csv`
- `reports/feature_importance_top20.png`

## Run API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Notebook
Run `bulldozer_end_to_end.ipynb` top-to-bottom.
