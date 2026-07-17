("""
Utility helpers for saving and loading models.
""")
import joblib
from pathlib import Path


def save_model(model, filepath):
	"""Save a model to `filepath` using joblib."""
	path = Path(filepath)
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, filepath)


def load_model(filepath):
	"""Load a joblib model from `filepath`."""
	return joblib.load(filepath)

