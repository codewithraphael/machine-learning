import joblib
from pathlib import Path

from config import MODELS_PATH


def save_best_model(trained_models, evaluation_results):
	"""Save the model with the strongest balanced multiclass performance."""
	if not trained_models or not evaluation_results:
		raise ValueError('trained_models and evaluation_results must not be empty')

	best_result = max(
		evaluation_results,
		key=lambda result: (
			result['Macro F1'],
			result['Balanced Accuracy'],
			result['Accuracy']
		)
	)
	best_name = best_result['Model']
	model_path = Path(MODELS_PATH) / 'best_multiclass_model.joblib'
	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(trained_models[best_name], model_path)

	print(
		f'\nSaved best multiclass model: {best_name}\n'
		f'Macro F1: {best_result["Macro F1"]:.3f}, '
		f'Balanced Accuracy: {best_result["Balanced Accuracy"]:.3f}, '
		f'Accuracy: {best_result["Accuracy"]:.3f}\n'
		f'Path: {model_path}'
	)

	return best_name, model_path

