import joblib

from config import MODELS_PATH

def save_model(model):

    joblib.dump(model, MODELS_PATH / 'sarima_model.joblib')