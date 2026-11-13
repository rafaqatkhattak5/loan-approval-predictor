# Loan Approval Predictor

A machine learning project that predicts loan approval from applicant details, served as a FastAPI app with a glass-style web UI.


## What I practiced
- Data cleaning: duplicates, dirty text, outliers (IQR)
- Missing values: median and mode, learned from train data only
- Encoding: ordinal (Education) and one-hot (other columns)
- Scaling: StandardScaler
- Models: Logistic Regression (75%) vs Random Forest (87%)

## Tech
Python, pandas, scikit-learn, FastAPI, ngrok, HTML/CSS/JS

## Run
1. `pip install -r requirements.txt`
2. `uvicorn app:app --port 8000`
3. Open `http://localhost:8000` (API docs at `/docs`)

## Note
The dataset was generated with code (synthetic data made for practice).
