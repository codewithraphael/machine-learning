# MAGIC Gamma Telescope Classification

This project uses machine-learning classifiers to distinguish gamma-ray events from hadronic background events in the MAGIC Gamma Telescope dataset.

## Dataset

The dataset contains 19,020 observations and 10 continuous input features. The file `data/magic04.data` is comma-separated and has no header row. The target column is `class`.

| Label | Meaning | Encoded value |
| --- | --- | ---: |
| `g` | Gamma-ray event (signal) | `1` |
| `h` | Hadron event (background) | `0` |

The input features are `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, and `fDist`. The original descriptions and scientific background are in [`data/magic04.names`](data/magic04.names).

The data was developed by the MAGIC project and is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).

## Project structure

```text
magic gamma telescope/
├── data/
│   ├── magic04.data
│   └── magic04.names
├── models/                  # Saved model pipelines
├── notebook/
│   └── eda.ipynb            # Exploratory data analysis
├── outputs/
│   └── plots/               # Evaluation plots
└── src/
    ├── config.py            # Data paths, features, and split settings
    ├── data_loader.py       # Loads the headerless data
    ├── data_cleaning.py     # Encodes the target
    ├── data_validation.py   # Prints dataset diagnostics
    ├── data_visualization.py
    ├── preprocessing.py    # Stratified train/validation/test split
    ├── train.py             # Builds and fits model pipelines
    ├── evaluation.py        # Metrics and plots
    ├── utils.py             # Model serialization
    └── main.py              # Project entry point
```

## Requirements

Use Python and install the project dependencies:

```powershell
python -m pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

## Run the project

From the `magic gamma telescope` directory, run:

```powershell
python src/main.py
```

The default configuration is in [`src/config.py`](src/config.py). The script loads the dataset, prints validation diagnostics, creates visualizations, splits the data, trains the models, reports evaluation results, saves plots, and serializes the XGBoost pipeline.

## Training workflow

1. Load the raw data and assign its documented column names.
2. Encode `g` as `1` and `h` as `0`.
3. Split the rows into stratified training, validation, and test sets. The default split is 70% / 15% / 15%.
4. Fit each classifier in a scikit-learn pipeline containing `StandardScaler`. The scaler is fitted on training data only and is applied automatically to validation and test inputs.
5. Train and evaluate the following classifiers:
   - Logistic regression
   - Random forest
   - K-nearest neighbors
   - Decision tree
   - XGBoost
   - Multilayer perceptron
6. Generate validation curves using five-fold stratified cross-validation on the training set. Each model's curve varies a model-specific hyperparameter and reports ROC-AUC.
7. Report test-set metrics, including accuracy, a classification report, a confusion matrix, and ROC-AUC where supported.

The test set is reserved for final evaluation and should not be used to select hyperparameters or decision thresholds.

## Evaluation files

The following plots are saved under `outputs/plots/`:

- `confusion_matrices_plot.png`
- `receiver_operating_characteristic_curves.png`
- `validation_curves_plot.png`

The trained XGBoost pipeline is saved to `models/xgb_model.joblib`. The saved pipeline includes its fitted scaler, so it can transform raw feature rows without a separate preprocessing artifact.

## Use the saved model

Load the saved pipeline and pass it a pandas DataFrame containing the ten input features in the order listed above:

```python
import joblib
import pandas as pd

from src.config import FEATURE_COLS, MODELS_DIR

model = joblib.load(MODELS_DIR / "xgb_model.joblib")
new_events = pd.read_csv("new_events.csv")
predictions = model.predict(new_events.loc[:, FEATURE_COLS])
```

Predictions use the same encoding as the training target: `1` for gamma and `0` for hadron. The model also supports `predict_proba` for class probabilities.

## References

- [MAGIC Gamma Telescope dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
- [`data/magic04.names`](data/magic04.names) for complete attribute definitions and dataset background
