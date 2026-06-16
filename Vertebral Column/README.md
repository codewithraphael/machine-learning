# Vertebral Column Dataset

Dataset file: vertebral_column.csv

## Overview

This dataset contains biomechanical measurements from patients' spines and a binary label indicating whether the case is considered abnormal. It is suitable for supervised learning experiments (binary classification) and exploratory analysis in biomechanics or medical ML tutorials.

## Files

- `vertebral_column.csv` — tabular CSV with numeric features and a binary target column `is_abnormal`.

## Column descriptions

Each column in `vertebral_column.csv` is numeric. Definitions below describe the typical meaning of each variable as used in biomechanics datasets; confirm exact measurement units in your original data source.

- `pelvic_incidence`: pelvic geometry measurement related to pelvic orientation.
- `pelvic_tilt`: angular measurement describing pelvic rotation.
- `lumbar_lordosis_angle`: curvature angle of the lumbar spine (lordosis).
- `sacral_slope`: angle of the sacral plate relative to horizontal.
- `pelvic_radius`: linear/positional measurement of the pelvis (dataset-specific units).
- `degree_spondylolisthesis`: severity/degree of vertebral slippage.
- `is_abnormal`: target label (1 = abnormal, 0 = normal).

Notes:
- All feature columns are continuous numeric values (float). Units may be degrees, millimeters, or dataset-specific—check the CSV source for exact units.

## Suggested use cases

- Binary classification: build models to predict `is_abnormal` (e.g., logistic regression, SVM, random forest, XGBoost).
- Model benchmarking: use as a small, well-scoped dataset to compare preprocessing pipelines and classifiers.
- Feature analysis: explore which biomechanical features correlate most strongly with abnormal cases (feature importance, correlation, SHAP).
- Anomaly detection / screening: treat `is_abnormal=1` as positive class for clinical screening experiments.

## Recommended preprocessing

- Inspect for missing values and outliers: handle or remove as appropriate.
- Scale or standardize features when using distance-based or gradient-based models (e.g., SVM, KNN, neural nets).
- Use stratified train/test splits if the classes are imbalanced.
- Consider cross-validation with stratification to obtain robust estimates.

## Citation and license

This dataset is commonly available from public repositories (for example, the UCI Machine Learning Repository). Verify the original source for citation and licensing details before publication or redistribution.