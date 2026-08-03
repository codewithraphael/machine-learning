import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models'

model = joblib.load(MODEL_PATH / 'best_model.joblib')

app = FastAPI()

class PeddictionRequest(BaseModel):
    



@app.get('/') # retrieve information from server
def root():
    return { 'message': 'api is live'}



@app.get('/about') # about project
def about():
    return {
        'project': 'bike demand prediction api',
        'version': '1.0'
    }



@app.post('/predict')
async def predict():
    pass