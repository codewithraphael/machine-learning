# Vertebral Column Dataset

Dataset file: `vertebral_column.csv`

## Overview
This dataset contains biomechanical features of patients' spines and a binary label indicating abnormality. It is commonly used for classification experiments in health/biomechanics and machine learning tutorials.

## Files
- `vertebral_column.csv` — tabular data (CSV) with numeric measurements and a binary target.

## Columns
- `pelvic_incidence` (float)
- `pelvic_tilt` (float)
- `lumbar_lordosis_angle` (float)
- `sacral_slope` (float)
- `pelvic_radius` (float)
- `degree_spondylolisthesis` (float)
- `is_abnormal` (int) — target label (1 = abnormal, 0 = normal)

These names are taken from the CSV header.

## Quick start (Python / pandas)
```python
import pandas as pd

df = pd.read_csv('vertebral_column.csv')
X = df.drop('is_abnormal', axis=1)
y = df['is_abnormal']

print(df.info())
print(df.describe())
```

## Suggested uses
- Binary classification (normal vs abnormal)
- Feature exploration and visualization
- Baseline models: logistic regression, decision trees, random forests, SVM

## Notes
- All features are numeric. Check for outliers and scale features as needed for some models.
- Confirm target balance before training; consider stratified splits or resampling if needed.

## Source & citation
This dataset is commonly distributed via public repositories (e.g., UCI Machine Learning Repository). If you use this dataset in published work, cite the original source or repository where you obtained the CSV.

## License
License information is not included with this file. Verify licensing from the original data source before reuse.
