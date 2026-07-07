import pandas as pd

import joblib
import  uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('../models/xgboost_model.joblib')

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
    
    return {'predicted_strength': prediction[0]}
