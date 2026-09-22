# Financial Risk Assessment

A machine learning project for predicting the credit risk level of loan applicants using structured financial and demographic features. The project is designed for a multiclass classification task where the target variable is the applicant's risk rating.

## Project Overview

This project uses a tabular dataset containing applicant information, financial health indicators, and loan characteristics to classify each record into one of the following risk categories:

- Low
- Medium
- High

The goal is to build a predictive model that can help assess whether a borrower represents a low, medium, or high lending risk.

## Problem Type

- Task: Multi-class classification
- Target variable: `Risk Rating`
- Output classes: `Low`, `Medium`, `High`

## Dataset

**File:** `data/financial_risk_assessment.csv`  
**Format:** CSV  
**Type:** Structured/tabular data  
**Use case:** Credit risk scoring and borrower classification

## Dataset Columns and Meaning

| Column Name | Type | Description |
|---|---|---|
| `Age` | Integer | Age of the applicant in years. |
| `Gender` | Categorical | Applicant gender (e.g., Male, Female, Non-binary). |
| `Education Level` | Categorical | Highest education level completed by the applicant (e.g., High School, Bachelor's, Master's, PhD). |
| `Marital Status` | Categorical | Current marital status such as Single, Married, Divorced, or Widowed. |
| `Income` | Float | Annual income of the applicant. |
| `Credit Score` | Integer/Float | Creditworthiness score representing the applicant's credit history quality. |
| `Loan Amount` | Float | Total amount requested or approved for the loan. |
| `Loan Purpose` | Categorical | Purpose of the loan, such as Auto, Business, Home, Personal, or others. |
| `Employment Status` | Categorical | Current employment status, e.g., Employed, Self-employed, or Unemployed. |
| `Years at Current Job` | Integer | Number of years the applicant has been in the current job. |
| `Payment History` | Categorical | Historical payment behavior rating, typically categorized as Excellent, Good, Fair, or Poor. |
| `Debt-to-Income Ratio` | Float | Ratio of monthly debt obligations to monthly income; a measure of financial burden. |
| `Assets Value` | Float | Total value of the applicant's assets. |
| `Number of Dependents` | Integer/Float | Number of people financially dependent on the applicant. |
| `City` | Categorical | City of residence. |
| `State` | Categorical | State or province where the applicant resides. |
| `Country` | Categorical | Country of residence. |
| `Previous Defaults` | Integer/Float | Number of previous loan defaults or missed payments the applicant has had. |
| `Marital Status Change` | Integer | Number of times the applicant's marital status has changed historically. |
| `Risk Rating` | Categorical | Target variable indicating the applicant's risk level: `Low`, `Medium`, or `High`. |

## Data Notes

- The dataset contains both numerical and categorical features.
- Some fields may have missing values, which should be handled during preprocessing.
- Categorical variables often require encoding before training machine learning models.
- Missing values, outliers, and class imbalance should be reviewed before building the final model.

## Project Structure

```text
financial risk assessment/
├── api/
│   └── app.py
├── data/
│   └── financial_risk_assessment.csv
├── evaluation result/
├── models/
├── notebook/
│   └── eda.ipynb
├── plots/
├── src/
│   ├── config.py
│   ├── data_cleaning.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Suggested Workflow

1. Load the dataset from `data/financial_risk_assessment.csv`
2. Inspect missing values and data types
3. Clean and transform the data
4. Encode categorical variables
5. Split the data into training and test sets
6. Train classification models
7. Evaluate using metrics such as accuracy, precision, recall, F1-score, and confusion matrix
8. Save the best-performing model in the `models/` directory

## Common Evaluation Metrics

For this type of task, the following metrics are commonly used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (if modeled as binary or one-vs-rest)

## Example Use Cases

- Automated loan approval screening
- Risk scoring for credit applicants
- Financial decision support for banks or lenders
- Early identification of high-risk borrowers

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Jupyter Notebook (optional, for analysis)

### Installation

```bash
pip install -r requirements.txt
```

### Run the project

Depending on the project implementation, you may run:

```bash
python src/main.py
```

or open and execute the notebook in the `notebook/` folder.

## Notes

This project is intended to illustrate a standard machine learning workflow for financial risk classification. It can be expanded with additional preprocessing, model comparison, and deployment components such as an API or a Streamlit dashboard.

---

This README provides a standard project summary and a clear explanation of each feature in the dataset so that the project is easy to understand and reproduce.
