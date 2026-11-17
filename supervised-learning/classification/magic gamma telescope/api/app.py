from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models' / 'xgb_model.joblib'

model = joblib.load(MODEL_DIR)

app = FastAPI()

class Item(BaseModel):
    pass