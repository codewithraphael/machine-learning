# Wine Quality Prediction Dataset

## Overview

The Wine Quality Dataset is a widely used dataset for machine learning and data analysis. It contains physicochemical properties of wine samples along with quality ratings assigned by human experts.

This dataset is commonly used to build models that predict wine quality based on measurable chemical features.

---

## Dataset Description

Each row in the dataset represents a wine sample, and each column describes a specific attribute of that wine.

### Features

| Feature              | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| type                 | Type of wine (Red or White)                                  |
| fixed acidity        | Non-volatile acids that remain stable                        |
| volatile acidity     | Acids that can evaporate (high values may indicate spoilage) |
| citric acid          | Adds freshness and flavor                                    |
| residual sugar       | Amount of sugar left after fermentation                      |
| chlorides            | Salt content in the wine                                     |
| free sulfur dioxide  | Active preservative component                                |
| total sulfur dioxide | Total sulfur content (free + bound)                          |
| density              | Density of the wine                                          |
| pH                   | Acidity level of the wine                                    |
| sulphates            | Preservative that also influences taste                      |
| alcohol              | Alcohol percentage                                           |
| quality              | Quality score assigned by experts (target variable)          |

---

## Objective

The main goal of this dataset is to build predictive models that estimate the quality of wine based on its chemical properties.

---

## Machine Learning Tasks

### Regression

Predict the exact wine quality score.

### Classification

Convert quality into categories such as:

* Good (quality ≥ 7)
* Average (quality = 5–6)
* Bad (quality ≤ 4)

---

## Typical Workflow

1. Data Cleaning
   Handle missing values and check for duplicates.

2. Feature Engineering
   Encode the type column and create new features if needed.

3. Data Preprocessing
   Scale numerical features and split data into training and testing sets.

4. Model Training
   Use models such as Linear Regression, Random Forest, and XGBoost.

5. Evaluation
   Use RMSE, MAE, or R² for regression and accuracy, precision, and recall for classification.

---

## Use Cases

* Predicting wine quality automatically
* Understanding factors that influence taste
* Practicing machine learning pipelines
* Feature importance analysis

---

## Getting Started

Clone the repository and install dependencies:

git clone [https://github.com/codewithraphael/wine-quality-project.git](https://github.com/codewithraphael/wine-quality-project.git)
cd wine-quality-project
pip install -r requirements.txt

