# Synthetic Heart Attack Dataset

This project uses a synthetic heart attack prediction dataset stored in [data/synthetic_heart_attack_dataset.csv](data/synthetic_heart_attack_dataset.csv). It is designed for binary classification tasks where the goal is to predict whether a patient is likely to experience a heart attack.

## Dataset Overview

- File: [data/synthetic_heart_attack_dataset.csv](data/synthetic_heart_attack_dataset.csv)
- Rows: 1000
- Columns: 14
- Target variable: `heart_attack`
- Missing values: 0
- Target distribution:
  - `1` (heart attack likely): 678
  - `0` (heart attack unlikely): 322

## Purpose of the Dataset

This dataset can be used to:
- Train machine learning models for heart attack prediction
- Explore how clinical and lifestyle features relate to heart attack risk
- Practice preprocessing, feature engineering, model training, and evaluation

> This dataset is synthetic and intended for educational and experimentation purposes. It should not be used as a substitute for professional medical advice or real clinical diagnosis.

## Column Descriptions

| Column | Meaning |
|---|---|
| `patient_id` | A unique identifier for each patient record. |
| `age` | The patient's age in years. |
| `sex` | Biological sex of the patient. Typical values are `Male` and `Female`. |
| `resting_bp` | Resting blood pressure measured in mmHg. |
| `cholesterol` | Serum cholesterol level, usually measured in mg/dL. |
| `fasting_bs` | Fasting blood sugar indicator. `0` usually means normal fasting blood sugar, while `1` indicates elevated fasting blood sugar. |
| `ecg_result` | Electrocardiogram result. Common values include `Normal`, `ST-T abnormality`, and `Left ventricular hypertrophy`. |
| `max_heart_rate` | Maximum heart rate achieved during exercise, measured in beats per minute. |
| `exercise_angina` | Whether the patient experienced angina during exercise. Values are typically `Y` (Yes) or `N` (No). |
| `st_depression` | ST depression induced by exercise relative to rest, often used as an indicator of stress on the heart. |
| `slope` | The slope of the ST segment during exercise testing. Common values are `Up`, `Flat`, and `Down`. |
| `num_major_vessels` | Number of major vessels visible on fluoroscopy or imaging, usually ranging from `0` to `3`. |
| `thalassemia` | A medical condition related to blood cells. Common categories include `Normal`, `Fixed Defect`, and `Reversible Defect`. |
| `heart_attack` | The target label. `1` indicates a positive case (heart attack likely), while `0` indicates a negative case. |

## Suggested Data Types

The dataset contains a mix of:
- Numeric features such as `age`, `resting_bp`, `cholesterol`, `max_heart_rate`, and `st_depression`
- Categorical features such as `sex`, `ecg_result`, `exercise_angina`, `slope`, `num_major_vessels`, and `thalassemia`
- Binary target variable: `heart_attack`

## Example of How the Data Can Be Used

A typical workflow would be:
1. Load the dataset
2. Clean and preprocess the data
3. Encode categorical variables
4. Split the data into training and testing sets
5. Train a classification model
6. Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC

## Recommended Preprocessing Steps

Before training a model, it is often helpful to:
- Convert categorical variables into numeric form
- Scale continuous variables if required by the model
- Check for class imbalance
- Split the data into train and test sets

## Notes

- `patient_id` should usually be removed from predictive modeling because it acts as an identifier rather than a meaningful feature.
- The target variable `heart_attack` is the main label to predict.
- Because the dataset is synthetic, model results may be easier to learn than with real-world medical datasets.
