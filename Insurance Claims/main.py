# %%
# ===== Loading Methods =====
from src.preprocessing import load_data, feature_selection, num_cat_split, preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model, predict_probabilities, validate

import joblib

import warnings
warnings.filterwarnings('ignore')


# %%
# ===== Loading Dataset =====
import argparse
import logging
from src.utils import save_model


def run_pipeline(save_models: bool = False):
	"""Run the full pipeline: load, preprocess, train, evaluate.

	Returns a dict with keys: df, preprocessor, X_train_prep, X_test_prep,
	y_train, y_test, trained_models
	"""
	# Load data
	df = load_data('data/processed/clean_insurance.csv')

	# Split features/target
	X_train, X_test, y_train, y_test = feature_selection(df)

	# Column lists + preprocessor
	num_cols, cat_cols = num_cat_split(df)
	preprocessor = preprocess_data(num_cols, cat_cols)

	# Fit and transform
	preprocessor.fit(X_train)
	X_train_prepared = preprocessor.transform(X_train)
	X_test_prepared = preprocessor.transform(X_test)

	# Train models
	trained_models = train_model(X_train_prepared, y_train)

	# Evaluate first model for quick feedback
	first_model = None
	if len(trained_models) > 0:
		first_model = list(trained_models.values())[0]
	if first_model is not None:
		acc, report = evaluate_model(first_model, X_test_prepared, y_test)
		print(f"Test accuracy: {acc:.4f}")
		print(report)

	# Optionally save models + preprocessor
	if save_models:
		for name, model in trained_models.items():
			save_model(model, f'models/{name}.joblib')
		save_model(preprocessor, 'models/preprocessor.joblib')
		logging.info('Saved models and preprocessor to models/')

	return {
		'df': df,
		'preprocessor': preprocessor,
		'X_train_prepared': X_train_prepared,
		'X_test_prepared': X_test_prepared,
		'y_train': y_train,
		'y_test': y_test,
		'trained_models': trained_models,
	}


# ===== Additional display / orchestration functions =====
import src.eda as eda_mod
import src.data_loader as data_loader_mod
import src.preprocessing as preprocessing_mod
import src.feature_engineering as fe_mod
import src.train as train_mod
import src.evaluate as evaluate_mod
import src.predict as predict_mod
import src.utils as utils_mod


def show_data_overview(df=None):
	"""Print basic data overview: head, info, and describe."""
	if df is None:
		df = clean_insurance
	print("---- Data head ----")
	print(df.head())
	print("\n---- Data info ----")
	df.info()
	print("\n---- Describe ----")
	print(df.describe(include='all'))


def run_eda_summary(df=None):
	"""Run EDA summary helpers from `src/eda.py` where available."""
	if df is None:
		df = clean_insurance
	try:
		eda_mod.summary_stat(df)
	except Exception as e:
		print("eda.summary_stat failed:", e)
	try:
		eda_mod.missing_values(df)
	except Exception as e:
		print("eda.missing_values failed:", e)


def run_preprocessing_preview(df=None):
	"""Show preprocessing column splits and a preprocessor preview."""
	if df is None:
		df = clean_insurance
	num_cols, cat_cols = preprocessing_mod.num_cat_split(df)
	print("Numeric cols:", num_cols)
	print("Categorical cols:", cat_cols)
	preproc = preprocessing_mod.preprocess_data(num_cols, cat_cols)
	print("Preprocessor created:", preproc)


def run_feature_engineering_preview(df=None):
	"""Run simple feature engineering preview (placeholder)."""
	if df is None:
		df = clean_insurance
	fe_df = fe_mod.engineer_features(df)
	print("Feature engineering returned dataframe with shape:", fe_df.shape)


def show_trained_models_metrics(trained_models_dict=None, X_test_prep=None, y_test_local=None):
	"""Evaluate and print metrics for trained models available in the module scope."""
	if trained_models_dict is None:
		trained_models_dict = trained_models
	if X_test_prep is None:
		X_test_prep = X_test_prepared
	if y_test_local is None:
		y_test_local = y_test
	for name, model in trained_models_dict.items():
		try:
			acc, report = evaluate_model(model, X_test_prep, y_test_local)
			print(f"Model: {name} - accuracy: {acc:.4f}")
			print(report)
		except Exception as e:
			print(f"Failed to evaluate {name}: {e}")


def show_all_results(pipeline_outputs=None):
	"""Convenience function that runs all small report functions.

	This will not retrain models; it uses variables already created
	earlier in the script (`clean_insurance`, `trained_models`, etc.).
	"""
	if pipeline_outputs is None:
		print("No pipeline outputs provided; nothing to show.")
		return

	df = pipeline_outputs.get('df')
	trained_models_dict = pipeline_outputs.get('trained_models')
	X_test_prep = pipeline_outputs.get('X_test_prepared')
	y_test_local = pipeline_outputs.get('y_test')

	print("Showing data overview...")
	show_data_overview(df)
	print("\nRunning EDA summaries...")
	run_eda_summary(df)
	print("\nPreprocessing preview...")
	run_preprocessing_preview(df)
	print("\nFeature engineering preview...")
	run_feature_engineering_preview(df)
	print("\nModel metrics...")
	show_trained_models_metrics(trained_models_dict, X_test_prep, y_test_local)


def main(argv=None):
	parser = argparse.ArgumentParser(description='Run insurance claims pipeline')
	parser.add_argument('--save-models', action='store_true', help='Save models and preprocessor to models/')
	parser.add_argument('--show-results', action='store_true', help='Show EDA and model results after running')
	args = parser.parse_args(argv)

	outputs = run_pipeline(save_models=args.save_models)
	if args.show_results:
		show_all_results(outputs)


if __name__ == '__main__':
	main()


if __name__ == "__main__":
	# When executed as a script, show the assembled reports.
	show_all_results()