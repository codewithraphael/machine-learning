import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_multiclass_model.joblib"


app = FastAPI()

model = joblib.load(MODEL_PATH)

class FinancialRiskAssessment(BaseModel):
    age: int
    gender: str
    education_level: str
    martial_status: str
    income: float
    credit_score: float
    loan_amount: float
    loan_purpose: str
    employment_status: str
    years_at_current_job: int
    payment_history: str
    debt_to_income_ratio: float
    assets_value: float
    number_of_dependents: int
    city: str
    state: str
    country: str
    previous_defaults: int
    marital_status_change: int


@app.get("/") # retrieve information from the server
def root():
    return{'message': 'api is running'}


@app.post("/predict") # send data or payload to the server
def predict(data: FinancialRiskAssessment):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = int(prob >= 0.5)

    return {
        'financial risk probability': round(float(prob), 4),
        'prediction': prediction,
        'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
    }