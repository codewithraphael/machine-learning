from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load('../stroke api/model/stroke_model.joblib')
THRESHOLD = 0.35

class PatientData(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    cardio_risk: int
    log_glucose: float
    bmi_category: str

@app.get("/")
def root():
    return{'message': 'stroke prediction api is running'}


@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.model_dump()
    input_df =pd.DataFrame([input_dict])

    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = int(prob >= THRESHOLD)

    return {
        'stroke probablity': round(float(prob), 4),
        'prediction': prediction,
        'result': 'High stoke risk' if prediction == 1 else 'Low stroke risk'
    }