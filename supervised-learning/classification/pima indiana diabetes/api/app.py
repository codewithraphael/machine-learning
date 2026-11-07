from fastapi import FastAPI
from pydantic import BaseModel


import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models' / 'logistic_regression_model.joblib'

model = joblib.load(MODEL_DIR)

app = FastAPI(title="Pima Indian Diabetes Classification API", version="1.0.0")

class PimaData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int



@app.get('/')
def root():
    return {"message": "Pima Indian Diabetes Classification API is Live"}


@app.post('/predict')
def predict(data: PimaData):
    input_data = [[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        data.age
    ]]
    prediction = model.predict(input_data)
   
    if prediction[0] == 1:
        return {"prediction": "Diabetic"}
    else:
        return {"prediction": "Non-Diabetic"}
    



