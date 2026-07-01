import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.joblib"


app = FastAPI()

# Testing Prediction Using Logistic Regression Model From Trained Models

model = joblib.load(MODEL_PATH)

class EngineCondition(BaseModel):
    engine_rpm: int
    lub_oil_pressure: float
    fuel_pressure: float
    coolant_pressure: float
    lub_oil_temp: float
    coolant_temp: float
    total_pressure: float
    total_temp: float
    temp_difference: float
    pressure_difference: float
    temp_ratio: float
    pressure_ratio: float
    rpm_oil_temp: float
    rpm_coolant_temp: float
    coolant_temp_per_rpm: float
    oil_temp_per_rpm: float
    pressure_std: float
    temp_std: float
    rpm_pressure: float
    rpm_squared: int
    engine_stress_index: float
    load_level: str
    oil_temp_category: str
    oil_pressure_status: str

'''
@app.get("/") # retrieve information from the server
def root():
    return{'message': 'api is running'}
'''

@app.post("/predict") # send data or payload to the server
def predict(data: EngineCondition):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = int(prob >= 0.5)

    return {
        'engine condition probability': round(float(prob), 4),
        'prediction': prediction,
        'result': 'Engine is in good condition' if prediction == 1 else 'Engine is in bad condition'
    }