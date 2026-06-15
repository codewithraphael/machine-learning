# %%
# ===== Prediction =====
def make_prediction(model, X):
	"""Return class predictions and probabilities for input `X` using `model`.

	Parameters
	----------
	model : estimator
		Trained sklearn-like model with `predict` and optionally `predict_proba`.
	X : array-like or DataFrame
		Input features.

	Returns
	-------
	tuple
		`(y_pred, y_prob)` where `y_prob` is `None` if `predict_proba` is unavailable.
	"""
	y_pred = model.predict(X)
	y_prob = None
	if hasattr(model, 'predict_proba'):
		try:
			y_prob = model.predict_proba(X)
		except Exception:
			y_prob = None

	return y_pred, y_prob
