# Credit Card Fraud Detection — Dataset & Task

## Overview

This README describes the `credit_card_fraud.csv` dataset and outlines a comprehensive, advanced workflow for building, evaluating, and deploying models to detect fraudulent credit card transactions. The guidance below is suited for reproducible experiments, rigorous evaluation for imbalanced classification, and production-ready pipelines.

## Dataset Summary

- File: `credit_card_fraud.csv`
- Typical size: ~284,000 rows (dataset variants may differ)
- Format: CSV
- Primary task: Binary classification (fraudulent vs legitimate transactions)

## Typical Columns

The canonical Kaggle credit card fraud dataset (commonly used) contains:

- `Time` — Seconds elapsed between this transaction and the first transaction in the dataset. Useful for temporal analysis and time-window features.
- `V1`–`V28` — Results of a PCA transformation applied to the original features (anonymized components). Treat these as numeric continuous features.
- `Amount` — Transaction amount. Requires scaling/normalization in many models.
- `Class` — Target: 0 = legitimate, 1 = fraud. This is highly imbalanced (frauds << non-frauds).

If your file has other or additional columns, add their descriptions here.

## Key Characteristics & Challenges

- Extreme class imbalance: fraud events are rare — use appropriate strategies for training and evaluation.
- Anonymized features (PCA components): limits direct interpretability; rely on model-agnostic explainers.
- Temporal patterns: fraud may exhibit time-dependent behavior — avoid leakage when splitting train/test.

## Recommended File Structure (for an experiment)

- `data/credit_card_fraud.csv` — raw CSV (source of truth)
- `notebooks/` — exploratory EDA notebooks
- `src/` — preprocessing, modeling, training, inference scripts
- `models/` — serialized model artifacts (`.joblib`, `.pkl`)
- `reports/` — figures, evaluation reports

## Exploratory Data Analysis (EDA)

1. Class balance: report counts and ratio of fraud vs legitimate. Use barplots and log-scale when helpful.
2. Transaction amount: distribution (histogram), log-transform, compare amounts by class.
3. Time-series checks: plot fraud rate over time (hour/day bins) to detect temporal clusters.
4. Feature distributions: inspect `V1`–`V28` distributions for anomalies and outliers.
5. Correlations: compute correlation / mutual information with the target. For PCA features, correlation patterns can still reveal signal clusters.

Visuals to produce: class-count bar chart, histograms for `Amount`, density plots per class, heatmap of correlations, precision-recall curve, ROC curve.

## Preprocessing Recommendations

- Missing values: canonical dataset has none, but validate and impute if present (median for continuous values).
- Scaling: apply `StandardScaler` or `RobustScaler` to `Amount` and `Time`-derived features. PCA features are already scaled.
- Time features: derive `hour_of_day`, `day_of_week`, rolling counts/amounts per card (if card id available).
- Feature selection: consider tree-based importance or recursive elimination; for PCA features, selection is less interpretable but still possible.
- Outlier handling: robust scaling or clipping for `Amount` can reduce influence of very large outliers.

## Handling Imbalance

- Resampling: experiment with `SMOTE`, `ADASYN`, or combined over-/under-sampling (e.g., SMOTE + Tomek links). Use resampling on training folds only.
- Class weights: prefer `class_weight='balanced'` in linear models and many scikit-learn estimators.
- Threshold tuning: optimize classification threshold for business metric (maximize recall at a minimum precision, or minimize cost).
- Ensemble and bagging with stratified sampling can help stabilize rare-event learning.

## Modeling Approaches (ordered from baseline → advanced)

1. Baseline: Logistic Regression with class weights and scaled numeric features.
2. Tree-based models: Random Forest, XGBoost, LightGBM, CatBoost (tune `scale_pos_weight` or class weights).
3. Calibrated models: Platt scaling or isotonic regression for probability calibration.
4. Anomaly detection approaches: Isolation Forest, One-Class SVM, Autoencoder (unsupervised or semi-supervised for rare events).
5. Deep learning: class-weighted cross-entropy, focal loss, or sequence models if temporal/sequential per-card data available.

## Evaluation Metrics & Strategy

- Primary metrics for imbalanced classification:
  - Precision, Recall, F1-score (report per-class and macro/micro averages)
  - Area under Precision-Recall curve (PR-AUC) — more informative than ROC-AUC when classes are imbalanced
  - ROC-AUC as complementary metric
  - Confusion matrix at chosen threshold(s)
  - Average precision

- Cost-aware evaluation: compute expected monetary cost (false positives cost vs false negatives cost). Use a cost matrix to pick operating point.

- Cross-validation:
  - Use `StratifiedKFold` to preserve class ratios.
  - If temporal leakage is possible, use temporal splits (time-based holdout) or expanding window validation.
  - Always apply resampling and scaling inside the CV fold pipeline to prevent leakage.

## Example scikit-learn Pipeline (concept)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42))
])

# Use StratifiedKFold CV and evaluate PR-AUC
```

## Hyperparameter Tuning

- Use `RandomizedSearchCV` / `BayesSearchCV` with `StratifiedKFold` for efficiency.
- Tune for metrics aligned with the business goal (e.g., maximize recall subject to precision >= X).

## Explainability & Monitoring

- Use SHAP to explain model predictions (global and local explanations).
- Track feature importance and drift: set up periodic checks for distributional shifts (e.g., PSI, population stability index).
- Monitor online metrics: predicted fraud rate, false positive rate, precision at threshold.

## Deployment & Inference

- Persist preprocessing pipeline + model together (e.g., `joblib.dump({'pipeline': pipeline}, 'models/model.joblib')`).
- For batch inference: run preprocessing and predict probabilities, then apply chosen threshold to flag transactions.
- For real-time inference: ensure low-latency model (light GBM or distilled model), and consistent preprocessing in the inference service.

## Reproducibility

- Record versions in `requirements.txt` or `environment.yml` and fix a `random_state` across experiments.
- Save model artifacts with metadata (training dataset hash, feature list, CV scores, training date).

## Suggested Experiments & Benchmarks

- Baseline logistic regression (class weights) → record PR-AUC and recall@precision=0.90.
- LightGBM with `scale_pos_weight` tuned + SMOTE on training folds.
- Autoencoder anomaly detector evaluated by ranking anomaly scores and computing PR-AUC.
- Threshold tuning by cost matrix to minimize expected monetary loss.

## Notebook / Report Outline

1. Data ingestion & validation
2. EDA with target stratification
3. Preprocessing pipeline construction
4. Baseline model training + evaluation
5. Advanced models + hyperparameter search
6. Model explainability (SHAP) and threshold selection
7. Final model serialization and simple inference demo

## References & Further Reading

- Kaggle credit card fraud detection dataset (canonical source)
- Fawcett, T. (2006). An introduction to ROC analysis.
- He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data.

---

If you want, I can also:

- generate a starter notebook with EDA and a baseline pipeline,
- create a `requirements.txt` with recommended packages,
- or implement a runnable training script and unit tests.

Please tell me which of the above you'd like next.
