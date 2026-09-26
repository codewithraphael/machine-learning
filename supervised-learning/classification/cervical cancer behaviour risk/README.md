# Cervical Cancer Behaviour Risk

A machine learning project for predicting cervical cancer risk based on behavioral, social, and psychological indicators. The goal is to build a classification model that estimates whether an individual is likely to be at risk (`ca_cervix`) using structured questionnaire-style features.

## Project Overview

This project uses a tabular dataset containing behavioral and psychosocial variables related to sexual risk, hygiene, social support, empowerment, and motivation. The target variable is a binary label indicating whether the subject is classified as having a cervical cancer risk.

## Problem Type

- Task: Binary classification
- Target variable: `ca_cervix`
- Output classes: `0` and `1`
- Use case: Early risk screening and decision support

## Objective

The objective of this project is to:

- understand the dataset and feature relationships
- preprocess the data for model training
- build and compare classification models
- evaluate model performance using standard metrics
- save the best-performing model for future prediction

## Dataset

**File:** `data/sobar-72.csv`  
**Format:** CSV  
**Type:** Structured/tabular dataset  
**Description:** Behavioral and health risk indicators used to predict cervical cancer risk.

### Target Variable

- `ca_cervix`: risk label (`0` or `1`)

## Dataset Features

The dataset contains behavioral and psychological variables such as:

| Feature | Description |
|---|---|
| `behavior_sexualRisk` | Sexual risk behavior score |
| `behavior_eating` | Eating habit-related behavior score |
| `behavior_personalHygine` | Personal hygiene score |
| `intention_aggregation` | Risk aggregation intention score |
| `intention_commitment` | Commitment-related intention score |
| `attitude_consistency` | Attitude consistency score |
| `attitude_spontaneity` | Spontaneity in attitudes |
| `norm_significantPerson` | Influence of significant persons |
| `norm_fulfillment` | Norm fulfillment score |
| `perception_vulnerability` | Perceived vulnerability |
| `perception_severity` | Perceived severity |
| `motivation_strength` | Motivation strength |
| `motivation_willingness` | Motivation willingness |
| `socialSupport_emotionality` | Emotional support score |
| `socialSupport_appreciation` | Appreciation-related support |
| `socialSupport_instrumental` | Instrumental social support |
| `empowerment_knowledge` | Knowledge-related empowerment |
| `empowerment_abilities` | Personal abilities empowerment |
| `empowerment_desires` | Desire-driven empowerment |

## Data Notes

- The data is numerical and structured.
- Features may require scaling or normalization depending on the model used.
- Class balance should be checked before training.
- Missing values, if present, should be handled during preprocessing.

## Project Structure

```text
cervical cancer behaviour risk/
├── api/
├── data/
│   └── sobar-72.csv
├── evaluation result/
├── model/
├── plots/
├── src/
│   ├── config.py
│   └── main.py
├── README.md
└── requirements.txt
```

## Suggested Workflow

1. Load the dataset from `data/sobar-72.csv`
2. Inspect data types, missing values, and target distribution
3. Perform preprocessing and feature engineering
4. Split the data into training and testing sets
5. Train and compare classification models
6. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix
7. Save the best-performing model in the `model/` folder

## Common Evaluation Metrics

For this classification problem, the following metrics are commonly used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## Typical Models

Suitable models for this task may include:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine
- Gradient Boosting

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Jupyter Notebook (optional)

### Installation

```bash
pip install -r requirements.txt
```

### Run the project

Depending on the implementation, the workflow may be run with:

```bash
python src/main.py
```

or through a notebook in the project directory.

## Example Use Cases

- Predict cervical cancer risk from behavioral indicators
- Support early screening decisions
- Aid medical research and risk analysis
- Build decision-support tools for clinical or educational applications

## Notes

This project is intended to demonstrate a standard machine learning workflow for healthcare risk prediction. It can be extended with additional preprocessing, hyperparameter tuning, model comparison, and deployment components such as an API or dashboard.

---

This README provides a clear summary of the project goal, dataset, workflow, and expected modeling approach so that the project is easy to understand and reproduce.
