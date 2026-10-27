# DVD Rental: SQL-to-ML Recommendation Project

A beginner-friendly, end-to-end project using the PostgreSQL `dvdrental` sample database. It demonstrates SQL exploration, data extraction, preprocessing, a popularity baseline, item-to-item collaborative filtering, evaluation, and recommendation generation.

## 1. What problem are we solving?

Given a customer's past film rentals, recommend films they have not rented yet. Each rental is treated as an implicit positive interaction: it tells us that the customer rented a film, but it does **not** prove that they liked it.

This is a learning project, not a production recommender. Rental history is sparse and biased by availability, so recommendations should be interpreted as suggestions.

## 2. Project structure

```text
dvdrental_ml_recommender/
├── data/                    # Generated CSV files (not committed)
├── models/                  # Generated model artifacts (not committed)
├── reports/                 # Generated evaluation reports (not committed)
├── sql/
│   ├── 01_schema_exploration.sql
│   ├── 02_business_analysis.sql
│   └── 03_recommender_dataset.sql
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── extract.py
│   ├── preprocess.py
│   ├── train.py
│   ├── recommend.py
│   ├── evaluate.py
│   └── main.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## 3. Requirements

- Python 3.10 or newer (Python 3.12 is suitable)
- PostgreSQL with the `dvdrental` sample database restored
- A terminal opened in this project directory

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install packages:
```bash
python -m pip install -r requirements.txt
```

## 4. Restore and configure the database

If `dvdrental.tar` is the PostgreSQL sample archive, restore it using pgAdmin's Restore option or the PostgreSQL command-line tools. A typical command is:

```bash
pg_restore -U postgres -d dvdrental -v "C:\path\to\dvdrental.tar"
```

Create your local environment file by copying `.env.example` to `.env`, then edit the connection settings. Never commit `.env`.

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=dvdrental
DB_USER=postgres
DB_PASSWORD=your_local_password
```

The database connection code reads these values from the environment. The application does not create or modify database tables.

## 5. Learn the SQL first

Open the scripts in `sql/` in pgAdmin or `psql` and run them in order:

1. `01_schema_exploration.sql`: inspect tables, columns, and keys.
2. `02_business_analysis.sql`: practise joins, aggregation, grouping, and window functions.
3. `03_recommender_dataset.sql`: understand the exact customer-film interaction dataset used by Python.

Run a query section at a time if your SQL editor does not support running the whole file.

## 6. Run the pipeline

From the project root:

```bash
python -m src.main
```

The pipeline:
1. extracts rental interactions from PostgreSQL,
2. saves a raw CSV snapshot,
3. prepares unique customer-film interactions,
4. trains an item-item collaborative-filtering recommender,
5. evaluates it with a per-customer temporal holdout and Recall@10,
6. saves the model and evaluation report.

The first run may take a little time depending on your computer and database size.

## 7. Ask for recommendations

After the pipeline has run:

```bash
python -m src.recommend --customer-id 1 --top-n 10
```

Replace `1` with a customer ID that exists in your database. Recommendations exclude films already rented by that customer. If the customer has no usable history, the model falls back to popular films.

## 8. Important modelling choices

- **Interaction:** one customer renting one film.
- **Duplicate rentals:** collapsed to one customer-film pair for model fitting.
- **Model:** cosine similarity between films based on customer rental histories.
- **Cold start:** customers with no known rental history receive popular films.
- **Evaluation:** Recall@10 on each eligible customer's latest held-out film interaction. This is a simple offline metric, not proof of business impact.
- **Leakage control:** evaluation splits each customer's interactions in time before building the training matrix.
- **Limitations:** rental does not equal satisfaction; this starter ignores film descriptions, ratings, actors, and changing inventory availability.

## 9. Suggested next improvements

1. Add genre/category features and a content-based recommender.
2. Compare Recall@5, Recall@10, MAP@K, and NDCG@K.
3. Add a popularity-only baseline and compare it with collaborative filtering.
4. Build a FastAPI endpoint for recommendations.
5. Add tests, logging, experiment tracking, and scheduled model refreshes.
6. Later, move to a larger dataset if you want more realistic recommendation evaluation.

## 10. Git safety

Before committing:
```bash
git init
git add .
git commit -m "Create DVD rental recommendation pipeline"
```

The `.gitignore` excludes secrets, virtual environments, generated data, and model artifacts.
