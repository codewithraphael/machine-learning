import joblib

from config import MODELS_DIR


def save_model(model, model_path=MODELS_DIR / 'xgb_model.joblib'):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
