from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models' / 'xgb_model.joblib'

model = joblib.load(MODEL_DIR)

app = FastAPI()

class MGT(BaseModel):
        
    fLength: float
    fWidth: float
    fSize: float
    fConc:  float
    fConc1: float
    fAsym: float
    fM3Long: float
    fM3Trans: float
    fAlpha: float
    fDist: float


@app.get("/")
def root():
    return {"message": "Magic Gamma Telescope Classification API is Live"}


@app.post("/predict")
def predict(item: MGT):

    data = pd.DataFrame([item.model_dump()])
    
    prediction = model.predict(data)
    
    return {"prediction": int(prediction[0])}
