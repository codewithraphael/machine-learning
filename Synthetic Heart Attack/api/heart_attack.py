import pandas as pd

import joblib
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('../models/gradient_boosting_classifier.joblib')

class HeartAttack(BaseModel):
    age: int
    sex: str
    resting_bp: int
    cholesterol: int
    fasting_bs: int
    ecg_result: str
    max_heart_rate: int
    exercise_angina: str
    st_depression: float
    slope: str
    num_major_vessels: int
    thalassemia: str



@app.get('/') # retrieve information from the server
def root():
    return {'message': 'heart attack api is live'}

@app.post('/predict') # send data or payload to the server
def predict(data: HeartAttack):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = int(prob >= 0.5)

    return {
        'heart attack probability': round(float(prob), 4),
        'prediction': prediction,
        'result': 'Heart attack un-likely' if prediction == 0 else 'Heart attack likely'
    }