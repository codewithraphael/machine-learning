import pandas as pd
import joblib
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR /'models' /'logistic_regression.joblib'


app = FastAPI()

model = joblib.load(MODEL_PATH)


class CervicalCancerPrediction(BaseModel):

   behavior_sexualrisk: int
   behavior_eating: int
   behavior_personalhygine: int
   intention_aggregation: int
   intention_commitment: int
   attitude_consistency: int
   attitude_spontaneity: int
   norm_significantperson: int
   norm_fulfillment: int
   perception_vulnerability: int
   perception_severity: int
   motivation_strength: int
   motivation_willingness: int
   socialsupport_emotionality: int
   socialsupport_appreciation: int
   socialsupport_instrumental: int
   empowerment_knowledge: int
   empowerment_abilities: int
   empowerment_desires: int


@app.get("/")
def root():
   return {'cervical cancer api is live'}


@app.post("/predict")
def predict(data: CervicalCancerPrediction):
   input_dict = data.model_dump()
   input_df = pd.DataFrame([input_dict])

   prediction = model.predict(input_df)[0]
   probabilities = model.predict_proba(input_df)[0]
   positive_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

   return {
      'prediction': int(prediction),
      'cervical cancer probability': round(positive_probability, 4),
      'result': 'Cervical Cancer Likely Present' if prediction == 1 else 'Cervical Cancer Not Present'
   }