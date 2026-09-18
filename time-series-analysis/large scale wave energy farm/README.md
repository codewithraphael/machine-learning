# Large Scale Wave Energy Farm Distribution Analysis

## Project Overview

This project focuses on **predicting and analyzing wave energy converter (WEC) power output** using machine learning regression models. The goal is to develop accurate predictive models that can forecast the total power generation from large-scale wave energy farms based on spatial and temporal wave characteristics.

Wave energy is a renewable energy source with significant potential. This project leverages real-world WEC data from two major Australian coastal regions to understand and predict power distribution patterns across multiple wave energy converters in a distributed farm setting.

---

## Dataset Description

### Data Source
The dataset is obtained from the **UCI Machine Learning Repository** and contains real-world measurements from large-scale wave energy farm installations.

### Dataset Structure

The project includes four CSV files representing different configurations:

| Dataset | Location | Converters | Measurement Points |
|---------|----------|------------|-------------------|
| `WEC_Perth_100.csv` | Perth, Australia | 100 WECs | 100 measurement points |
| `WEC_Perth_49.csv` | Perth, Australia | 49 WECs | 49 measurement points |
| `WEC_Sydney_100.csv` | Sydney, Australia | 100 WECs | 100 measurement points |
| `WEC_Sydney_49.csv` | Sydney, Australia | 49 WECs | 49 measurement points |

### Features

Each dataset contains the following feature types:

#### 1. **Spatial Coordinates** (200 features for 100-converter datasets)
   - **X1-X100**: X-coordinates of wave energy converters
   - **Y1-Y100**: Y-coordinates of wave energy converters
   - These coordinates represent the physical locations of WECs in the farm

#### 2. **Individual WEC Power Output** (100 features for 100-converter datasets)
   - **Power1-Power100**: Individual power output from each wave energy converter
   - Measured in kilowatts (kW)

#### 3. **Aggregate Energy Metrics**
   - **qW**: Reactive power or quadrature component of power
   - **Total_Power**: Total power output from all converters in the farm (TARGET VARIABLE)

### Sample Statistics
- **Dataset Shape**: ~1000+ samples per configuration with 300+ features
- **Target Variable**: Total_Power (continuous values)
- **Feature Type**: Numerical/Float values
- **Data Characteristics**: Real-world sensor measurements with natural variability

---

##  Project Objectives

1. **Power Prediction**: Develop models to accurately predict total farm power output from individual WEC characteristics and spatial information
2. **Feature Analysis**: Understand which WECs and spatial features are most influential in power generation
3. **Model Comparison**: Evaluate and compare different regression algorithms for optimal performance
4. **Scalability Analysis**: Analyze differences between 49-converter and 100-converter farm configurations
5. **Geographic Comparison**: Compare model performance across different Australian coastal regions (Perth vs Sydney)

---

## Project Structure

```
large-scale-wave-energy-farm/
├── data/
│   └── WEC/
│       ├── WEC_Perth_100.csv        # Perth farm with 100 converters (training)
│       ├── WEC_Perth_49.csv         # Perth farm with 49 converters (testing)
│       ├── WEC_Sydney_100.csv       # Sydney farm with 100 converters (training)
│       └── WEC_Sydney_49.csv        # Sydney farm with 49 converters (testing)
├── src/
│   ├── __init__.py
│   ├── train/
│   │   ├── wec_perth_100.py        # Train models on Perth 100-converter dataset
│   │   ├── wec_sydney_100.py       # Train models on Sydney 100-converter dataset
│   │   └── config.py               # Configuration and utility functions
│   └── test/
│       ├── wec_perth_49.py         # Evaluate on Perth 49-converter dataset
│       └── wec_sydney_49.py        # Evaluate on Sydney 49-converter dataset
├── models/
│   └── (Trained model artifacts saved here)
├── plots/
│   └── (Visualization outputs)
├── evaluation result/
│   └── (Model performance metrics)
└── README.md
```

---

### Libraries
- **Pandas & NumPy**: Data manipulation and numerical computations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting regression models
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization and persistence

---


## 📈 Modeling Approach

### Algorithms Used

#### 1. **Linear Regression**
   - Simple baseline model
   - Interpretable coefficients for feature importance
   - Good for understanding linear relationships

#### 2. **XGBoost Regressor**
   - Gradient boosting ensemble method
   - Captures non-linear patterns in power generation
   - Handles feature interactions effectively
   - Expected to provide superior predictive performance

### Model Pipeline

```
Raw Data
   ↓
[EDA & Data Exploration]
   ↓
[Data Preprocessing]
   ├─→ Missing Value Handling
   ├─→ Feature Scaling (StandardScaler/MinMaxScaler)
   └─→ Train-Test Split
   ↓
[Model Training]
   ├─→ Linear Regression
   └─→ XGBRegressor
   ↓
[Model Evaluation]
   ├─→ R² Score
   ├─→ Mean Squared Error (MSE)
   ├─→ Mean Absolute Error (MAE)
   └─→ Root Mean Squared Error (RMSE)
   ↓
[Cross-Validation] (KFold)
   ↓
[Model Persistence & Visualization]
```

### Cross-Validation Strategy
- **K-Fold Cross-Validation** for robust performance estimation
- Helps prevent overfitting and provides more reliable metrics

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained (0-1, higher is better) |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Root average squared error (same units as target) |
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Average absolute prediction error (interpretable) |
| **MSE** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Mean squared error (penalizes large errors) |

---


### Training Models

#### Train on Perth 100-Converter Data
```bash
python src/train/wec_perth_100.py
```
- Loads `WEC_Perth_100.csv`
- Performs exploratory data analysis
- Trains regression models
- Generates visualizations
- Saves model artifacts to `models/`

#### Train on Sydney 100-Converter Data
```bash
python src/train/wec_sydney_100.py
```
- Same pipeline as Perth training
- Uses Sydney location data

### Testing & Evaluation

#### Evaluate on Perth 49-Converter Test Set
```bash
python src/test/wec_perth_49.py
```
- Loads trained models from `models/`
- Evaluates on Perth 49-converter data
- Generates performance metrics
- Creates evaluation visualizations

#### Evaluate on Sydney 49-Converter Test Set
```bash
python src/test/wec_sydney_49.py
```
- Evaluates models on Sydney 49-converter data

---

## Key Findings & Insights

(To be populated after model training and evaluation)

### Expected Observations
- **Model Comparison**: XGBoost likely outperforms Linear Regression due to non-linear relationships
- **Feature Importance**: Nearby WEC power outputs are stronger predictors than distant ones
- **Geographic Differences**: Perth and Sydney models may have different performance characteristics
- **Scalability**: 100-converter farm models may achieve better predictions than 49-converter models
- **Power Distribution**: Total power output shows non-uniform distribution across the farm

---

## Data Preprocessing Steps

### 1. **Missing Value Handling**
   - Check for NaN values in each feature
   - Implement imputation strategy (mean, median, or forward-fill)

### 2. **Feature Scaling**
   - StandardScaler: Zero mean, unit variance (for Linear Regression, distance-based methods)
   - MinMaxScaler: Normalize to [0, 1] range (for gradient boosting)

### 3. **Train-Test Split**
   - 80-20 or 75-25 split
   - Stratification may be applied if target distribution is skewed

### 4. **Dimensionality Considerations**
   - High-dimensional dataset (300+ features)
   - Feature selection or dimensionality reduction may improve model efficiency
   - Correlation analysis to identify redundant features

---

## Machine Learning Concepts Applied

1. **Regression Analysis**: Continuous value prediction
2. **Ensemble Methods**: XGBoost for improved predictions
3. **Cross-Validation**: K-Fold for robust performance estimation
4. **Feature Scaling**: Standardization for algorithm effectiveness
5. **Exploratory Data Analysis**: Understanding data distributions and relationships
6. **Model Evaluation**: Comprehensive metrics for performance assessment
7. **Hyperparameter Tuning**: (Can be extended with GridSearchCV/RandomizedSearchCV)

---

## Future Enhancements

1. **Hyperparameter Optimization**: Use GridSearchCV or Optuna for XGBoost tuning
2. **Feature Engineering**: Create interaction features and derived metrics
3. **Dimensionality Reduction**: Apply PCA or feature selection techniques
4. **Time-Series Analysis**: If timestamp data is available, include temporal patterns
5. **Deep Learning**: Explore neural networks (LSTM, MLP) for complex patterns
6. **Model Interpretability**: SHAP values for feature importance analysis
7. **Ensemble Methods**: Combine multiple models (Voting, Stacking)
8. **Real-Time Prediction**: Deploy model as REST API for live predictions
9. **Geographic Analysis**: Create location-based prediction models
10. **Energy Efficiency Analysis**: Correlate predictions with energy costs and efficiency

---

## References & Resources

### Dataset
- **UCI Machine Learning Repository**: [Wave Energy Farm Dataset](https://archive.ics.uci.edu/ml/)
- Publicly available datasets for renewable energy research

### Wave Energy
- International Energy Agency (IEA) Ocean Energy Reports
- Wave Energy Converter Technology Overview
- Renewable Energy Integration and Grid Management

---

This project demonstrates a comprehensive machine learning workflow for renewable energy prediction:
- Real-world data analysis and preprocessing
- Multiple regression algorithms comparison
- Rigorous model evaluation and validation
- Visualization and interpretation of results

The models developed here can be extended to:
- Online learning with streaming data
- Real-time power forecasting
- Farm optimization and control strategies
- Resource planning for energy distributors
