import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('../models/booking_model.joblib')

class InputData(BaseModel):
    features: list[float]

@app.get('/')
def root():
    return{'status': 'model is live'}

@app.post('/predict')
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {'prediction': prediction.tolist()}