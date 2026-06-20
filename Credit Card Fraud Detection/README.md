# 💳 Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using multiple classification models.

## 📋 Dataset Overview

**File**: `data/credit_card_fraud.csv`  
**Format**: CSV (comma-separated)  
**Primary Task**: Binary Classification (Fraud Detection)  
**Problem Type**: Imbalanced Classification

### Dataset Statistics

- **Total Records**: 10,000+ transactions
- **Features**: 9 input features + 1 target
- **Target Variable**: `is_fraud` (0 = Legitimate, 1 = Fraudulent)
- **Class Distribution**: Imbalanced (fraud cases are minority)

---

## 📊 Feature Description

| Feature | Type | Description | Range/Examples |
|---------|------|-------------|-----------------|
| `transaction_id` | String | Unique transaction identifier | TXN000001, TXN000002, ... |
| `amount` | Float | Transaction amount in dollars | 0.01 - 100,000+ |
| `merchant_category` | Categorical | Type of merchant/vendor | Fuel, Grocery, Restaurant, Entertainment, Utilities, Healthcare, Travel, Online Shopping |
| `transaction_hour` | Integer | Hour of day when transaction occurred | 0-23 (midnight to 11 PM) |
| `customer_age` | Integer | Age of the cardholder | 18-80+ years |
| `account_balance` | Float | Customer's account balance at time of transaction | Positive values ($) |
| `is_foreign_transaction` | Binary | Whether transaction is from outside home country | 0 (No), 1 (Yes) |
| `is_weekend` | Binary | Whether transaction occurred on weekend | 0 (Weekday), 1 (Weekend) |
| `num_transactions_today` | Integer | Number of transactions made by customer today | 1-50+ |
| `is_fraud` | Binary | **TARGET**: Whether transaction is fraudulent | 0 (Legitimate), 1 (Fraudulent) |

---

## 🎯 Key Characteristics & Challenges

### 1. **Class Imbalance**
- Fraudulent transactions are much rarer than legitimate ones
- Requires specialized handling: SMOTE, class weights, adjusted thresholds
- Standard accuracy metric is misleading; use Precision-Recall-AUC, F1-score, and ROC-AUC

### 2. **Interpretable Features**
- Unlike PCA-transformed datasets, these features are directly interpretable
- Patterns can be identified: unusual hours, foreign transactions, high amounts, etc.
- Enables explainability and feature engineering

### 3. **Behavioral Patterns**
- Fraud often exhibits temporal patterns (specific hours, unusual frequency)
- International transactions increase fraud risk
- Unusual transaction amounts relative to account balance are red flags
- Multiple rapid transactions may indicate compromised account

### 4. **Business Impact**
- False negatives (missing fraud): Financial loss, customer impact
- False positives (blocking legitimate transactions): Customer experience degradation
- Decision threshold should align with business cost-benefit analysis

---

## 📁 Project Structure

```
Credit Card Fraud Detection/

├── data/
│   └── credit_card_fraud.csv         # Raw dataset
├── models/
│   ├── logistic_regression_model.joblib          # Saved models
│   ├── decision_tree_classifier_model.joblib
│   ├── randomforest_classifier_model.joblib
│   └── xgboost_classifier_model.joblib
├── src/
│   ├── main.py                       # Training pipeline
├── plots/                            # Generated visualizations
└── streamlit/                        #Streamlit web application
|    └── app.py
├── README.md                         # This file
├── requirements.txt                  # Python dependencies                     
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# 1. Clone/navigate to project directory
cd "Credit Card Fraud Detection"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Training a Model

```bash
# Run the complete training pipeline
cd src
python main.py
```

---

## 🤖 Implemented Models

Four machine learning models are trained and evaluated:

| Model | Type | Strengths | Best For |
|-------|------|-----------|----------|
| **Logistic Regression** | Linear | Interpretable, fast, probabilistic | Baseline comparisons |
| **Decision Tree** | Tree-based | Simple, no scaling needed, feature importance | Explainability |
| **Random Forest** | Ensemble | Robust, handles imbalance well, non-linear patterns | Production reliability |
| **XGBoost** | Gradient Boosting | High accuracy, excellent imbalance handling, fast | Best performance |

---

## 📈 Exploratory Data Analysis (EDA)

### Key Analyses Performed

1. **Class Distribution**
   - Count of fraudulent vs legitimate transactions
   - Imbalance ratio and sampling strategies

2. **Feature Distributions**
   - Transaction amount statistics and outliers
   - Age, balance, and hourly patterns
   - Merchant category frequency

3. **Fraud Patterns**
   - Peak fraud hours (identify temporal anomalies)
   - Foreign transaction fraud rate
   - Amount-based fraud indicators
   - Velocity fraud (multiple transactions/day)

4. **Correlations**
   - Feature-target relationships
   - Inter-feature correlations
   - Feature importance for tree models

5. **Visualizations Generated**
   - Class distribution bar chart
   - Transaction amount histograms
   - Fraud rate by hour heatmap
   - Correlation matrices
   - ROC and Precision-Recall curves
   - Confusion matrices

---

## 🔧 Data Preprocessing Pipeline

### Steps Applied

1. **Cleaning**
   - Handle missing values (forward fill, mean imputation)
   - Remove duplicates
   - Validate data types

2. **Encoding**
   - One-hot encode `merchant_category` (converts to binary columns)
   - Keep binary features as-is (`is_foreign_transaction`, `is_weekend`)

3. **Scaling**
   - StandardScaler applied to numeric features:
     - `amount`
     - `customer_age`
     - `account_balance`
     - `transaction_hour`
     - `num_transactions_today`

4. **Feature Engineering**
   - Amount-to-balance ratio (spending relative to account)
   - Hour categorization (early morning, daytime, evening, night)
   - Merchant category frequency features

5. **Train-Test Split**
   - 80% training / 20% testing
   - Stratified split to maintain class distribution

---

## 📊 Model Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall correctness (use with caution due to imbalance)
- **Precision**: False positive rate (cost of blocking legitimate transactions)
- **Recall**: False negative rate (cost of missing fraud)
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Overall discriminative ability
- **PR-AUC**: Precision-Recall Area Under Curve (more informative for imbalanced data)

### Evaluation Strategy
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Class Weights**: Applied to handle imbalance (`balanced` mode)
- **Threshold Optimization**: Tuned for business metric (maximize recall at acceptable precision)

---

## 🎨 Streamlit Application

Interactive web interface for real-time fraud detection predictions.

### Features
- **Model Selection**: Switch between 4 trained models
- **Input Interface**: Enter transaction details (amount, category, hour, age, balance, etc.)
- **Real-time Prediction**: Instant fraud probability and classification
- **Risk Indicators**: Identifies specific fraud risk factors
- **Feature Guide**: Documentation of each input feature

### How to Use
1. Select a trained model from the sidebar
2. Enter transaction details in the form
3. Click "Analyze Transaction"
4. View prediction result and fraud probability
5. Review identified risk factors

---

---

## 💡 Usage Examples

### Making Predictions via Python

```python
import joblib
import numpy as np

# Load model
model = joblib.load('models/xgboost_classifier_model.joblib')

# Prepare features [amount, hour, age, balance, foreign, weekend, daily_txns]
transaction = np.array([[1500, 14, 45, 25000, 0, 0, 3]])

# Predict
prediction = model.predict(transaction)
probability = model.predict_proba(transaction)

print(f"Fraud: {prediction[0]}")
print(f"Fraud Probability: {probability[0][1]:.2%}")
```

### Using the Streamlit App

```bash
streamlit run app.py
# Navigate to http://localhost:8501
# Fill in transaction details
# Click "Analyze Transaction"
```

---

## 🔍 Fraud Indicators

Common fraud patterns detected by the model:

- **High Transaction Amount**: Large purchases relative to account balance
- **International Transactions**: Cross-border activities from unusual locations
- **Unusual Hours**: Transactions at odd hours (2 AM - 6 AM)
- **Rapid Velocity**: Multiple transactions within short timeframe
- **Unusual Categories**: Specific merchant categories with higher fraud rates
- **Account Age Patterns**: Fraud correlates with specific age groups

---

## 📦 Requirements

See `requirements.txt` for full dependencies:
- pandas - Data manipulation
- scikit-learn - Machine learning models
- xgboost - Gradient boosting
- joblib - Model serialization
- streamlit - Web framework
- numpy - Numerical computing

---

## 🛠️ Advanced Usage

### Retraining Models
```bash
python src/main.py
```

### Custom Threshold Tuning
Edit `src/config.py` to adjust classification thresholds based on business requirements.

### Adding New Features
Modify `src/feature_engineering.py` to create additional features.

---

## 📝 Notes

- **Data Privacy**: This is a synthetic or publicly available dataset for educational purposes
- **Real-World Deployment**: Consider adding real-time data validation, monitoring, and retraining pipelines
- **Model Monitoring**: Track drift in fraud patterns and model performance over time
- **Compliance**: Ensure compliance with financial regulations (PCI-DSS, etc.)

---

## 👤 Contributing

To improve this project:
1. Analyze edge cases and model failures
2. Experiment with new features
3. Test additional models or ensemble methods
4. Improve Streamlit UI/UX
5. Add automated testing and CI/CD

---

## 📄 License

See LICENSE file in the repository.

---

## 📧 Questions?

For issues or improvements, open an issue or contact the project maintainer.
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