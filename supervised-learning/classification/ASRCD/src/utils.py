import joblib

from config import MODEL_DIR


def save_model(trained_models):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for name, model in trained_models.items():
        model_path = MODEL_DIR / f'{name}.joblib'
        joblib.dump(model, model_path)
        saved_paths[name] = model_path

    return saved_paths
