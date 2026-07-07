from pathlib import Path

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'xgboost_model.joblib'
model = joblib.load(MODEL_PATH)

class ConcreteStrength(BaseModel):
    cement: float
    blast_furnace_slag: float
    fly_ash: float
    water: float
    superplasticizer: float
    coarse_aggregate: float
    fine_aggregate: float
    age: int


@app.get('/')
def root():
    return {'message': 'concrete strength prediction api is live'}


@app.post('/predict')
def predict(data: ConcreteStrength):
    input_dict = data.model_dump()
    input_data = pd.DataFrame([input_dict])
    prediction = model.predict(input_data)

    predicted_strength = float(prediction[0])
    
    return {'predicted_strength': predicted_strength}
