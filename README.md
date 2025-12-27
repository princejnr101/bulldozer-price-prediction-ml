# Bluebook for Bulldozers — End-to-End Regression (RMSLE)

End-to-end regression pipeline using the Kaggle **Bluebook for Bulldozers** dataset.
Includes: preprocessing, time-based split, baseline + tuned RandomForest, model persistence,
test predictions, feature importance, and a FastAPI deployment.

## Project Goal
Predict **SalePrice** (regression). Primary metric: **RMSLE**.

---

## Data (do NOT commit to GitHub)
Put Kaggle CSVs in either location:

**Option A (recommended):**
- `data/TrainAndValid.csv`
- `data/Test.csv`

**Option B:**
- `TrainAndValid.csv` and `Test.csv` next to the notebook/script

Loader also handles common variants like `TrainAndValid (1).csv`.

---

## Setup
```bash
pip install -r requirements.txt


Train + Tune
python train.py


Outputs (local):

models/model.joblib (trained pipeline)

models/metrics.json (baseline vs tuned + best params)

reports/feature_importance.csv

reports/feature_importance_top20.png

Note: models/model.joblib can be very large and is intentionally not pushed to GitHub.


Run the Notebook

Run bulldozer_end_to_end.ipynb top-to-bottom.

Run API (FastAPI)
uvicorn app:app --host 0.0.0.0 --port 8000

Health Check
curl http://localhost:8000/health

Predict (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "saledate": "2012-11-19",
    "features": {
      "MachineID": 999999,
      "ModelID": 4605,
      "fiBaseModel": 555,
      "YearMade": 2005,
      "ProductSize": "Medium",
      "state": "Florida",
      "UsageBand": "High"
    }
  }'

Deployment Note (Render)

This repo includes app.py, requirements.txt, and Procfile for Render.
Because the model file is large, you must provide models/model.joblib to the deployed service
(e.g., download from external storage at startup or train during build).


### Now update GitHub (exact commands)
1) Open README in notepad:
```powershell
notepad README.md


Paste the clean content above → Save → Close.

Commit and push:

git add README.md
git commit -m "Update README with usage and deployment notes"
git push
