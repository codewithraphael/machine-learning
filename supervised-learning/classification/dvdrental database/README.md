# DVD Rental: Customer Rental Prediction

This project uses the PostgreSQL `dvdrental` sample database to build a
binary classification model that predicts whether a customer is likely to
rent another movie within 30 days of a rental.

It is a small, beginner-friendly machine-learning project covering SQL
feature extraction, data cleaning, preprocessing, model training, and
evaluation.

## 1. Prediction target

Each training row represents one customer rental. The target column is:

```text
rented_again_within_30_days
```

- `1`: the customer rented another movie within 30 days.
- `0`: the customer did not rent another movie within 30 days.

The feature query excludes rentals from the final 30 days of the database
history because their complete outcome window is not available. Features such
as prior rental count and time since the previous rental are calculated from
rentals before the current rental to reduce future-data leakage.

This model estimates repeat-rental likelihood from historical behavior. It
does not prove that a customer will rent again, and it should not be treated
as a production decision system without further validation.

## 2. Project structure

```text
dvdrental database/
├── data/
│   ├── customer_rental_prediction.csv  # Generated or exported training data
│   └── dvdrental.tar                   # PostgreSQL sample database archive
├── models/                             # Saved model artifacts
├── notebooks/                          # Exploratory notebooks
├── outputs/                            # Evaluation results and predictions
├── queries/
│   ├── analysis.sql                    # SQL analysis examples
│   ├── exploration.sql                 # Schema and data exploration
│   └── feature_query.sql               # Customer rental feature dataset
├── src/
│   ├── config.py                       # Environment and query configuration
│   ├── data_cleaning.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── .env
└── requirements.txt
```

## 3. Requirements

- Python 3.10 or newer
- PostgreSQL with the `dvdrental` database restored
- A terminal opened in this project directory

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 4. Restore and configure PostgreSQL

Restore `data/dvdrental.tar` into a local PostgreSQL database using pgAdmin or
`pg_restore`:

```powershell
pg_restore -U postgres -d dvdrental -v "C:\path\to\dvdrental.tar"
```

Configure the connection values in `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=dvdrental
DB_USER=postgres
DB_PASSWORD=your_local_password
```

Do not commit `.env` or any real database credentials.

## 5. Generate the training dataset

Run [feature_query.sql](queries/feature_query.sql) in pgAdmin or `psql`.
Export the result as:

```text
data/customer_rental_prediction.csv
```

The query joins customer, rental, inventory, and film data and produces
features including:

- Customer and rental identifiers
- Store and film information
- Film rating, length, rental duration, and rental rate
- Rental weekday and hour
- Number of previous rentals
- Days since the previous rental
- Average historical time between rentals
- The 30-day repeat-rental target

Run the scripts in [exploration.sql](queries/exploration.sql) and
[analysis.sql](queries/analysis.sql) first if you want to inspect the schema
and rental data.

## 6. Train and evaluate the model

The source modules provide the project workflow:

1. Load the exported CSV or query the PostgreSQL database.
2. Validate columns and data types.
3. Clean missing and invalid values.
4. Transform categorical and numeric features.
5. Train a binary classification model.
6. Evaluate predictions on held-out data.
7. Save the model and evaluation outputs.

Run the relevant module from the project directory, for example:

```bash
python -m src.train
python -m src.evaluate
```

If your local implementation exposes a different entry point, use the
corresponding script in `src/`.

## 7. Recommended evaluation metrics

Because repeat rentals may be imbalanced, do not rely on accuracy alone.
Review:

- Precision: how many predicted repeat renters actually rented again.
- Recall: how many repeat renters the model identified.
- F1 score: balance between precision and recall.
- ROC-AUC: ranking quality across classification thresholds.
- PR-AUC: useful when the positive class is relatively small.
- Confusion matrix: counts of true and false predictions.

Choose the probability threshold according to the use case. For example, a
retention campaign may prefer higher recall, while a costly promotion may
prefer higher precision.

## 8. Modelling considerations

- Use a time-aware train/test split where possible.
- Do not use future rentals, future payments, or post-rental outcomes as
  features.
- Fit preprocessing steps only on the training set.
- Compare the model with a simple baseline, such as always predicting the
  majority class.
- Check performance across stores and customer activity levels.
- Treat a rental as observed behavior, not proof of customer satisfaction.

## 9. Possible improvements

1. Add rolling features for recent rental activity.
2. Include payment totals and return behavior when they are available before
   the prediction point.
3. Compare logistic regression, random forest, and gradient-boosting models.
4. Tune the classification threshold for a specific retention campaign.
5. Add cross-validation that preserves chronological order.
6. Track model performance as new rental data becomes available.

## 10. Git and data safety

Before committing:

```bash
git add .
git commit -m "Document customer rental prediction project"
```

Keep credentials, private exports, generated model artifacts, and other
sensitive data out of version control.
