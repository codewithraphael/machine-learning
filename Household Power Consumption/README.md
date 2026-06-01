# Household Power Consumption Analysis & Forecasting

## Project Overview

This project provides a **comprehensive, multi-phase deep dive** into household electricity consumption patterns using the UCI Machine Learning Repository's Individual Household Electric Power Consumption dataset. It spans from foundational exploratory analysis through advanced machine learning and deep learning approaches, demonstrating progressively sophisticated techniques for time-series forecasting, anomaly detection, and load profiling.

**Dataset:** 4 years (Dec 2006 - Nov 2010) of minute-level electrical consumption data from a French household  
**Records:** ~2.1M observations  
**Frequency:** 1-minute intervals

---

## Dataset Description

### Columns & Meanings

| Column | Unit | Description |
|--------|------|-------------|
| **Date** | YYYY-MM-DD | Observation date |
| **Time** | HH:MM:SS | Observation time (24-hour format) |
| **Global_active_power** | kW | Household total active power consumption |
| **Global_reactive_power** | kVAR | Household total reactive power |
| **Voltage** | V | Household voltage |
| **Global_intensity** | A | Household total current intensity |
| **Sub_metering_1** | Wh | Energy sub-metering (Kitchen: dishwasher, oven, microwave) |
| **Sub_metering_2** | Wh | Energy sub-metering (Laundry: washer, dryer, water-heater) |
| **Sub_metering_3** | Wh | Energy sub-metering (Electric water-heating & air-conditioning) |

### Key Characteristics
- **Missing Data:** ~1.25% (marked as '?') → forward-filled for continuity
- **Time Resolution:** 1-minute samples (can be resampled to hourly/daily)
- **Target Variable:** `Global_active_power` (total household consumption)
- **Temporal Patterns:** Strong daily, weekly, and seasonal cycles

---

## Project Structure

The analysis is organized into **5 progressive phases**, each building on the previous:

```
Household Power Consumption/
├── 01_Explanatory_Data_Analysis/          # Phase 1: Foundation
├── 02_Statistical_Baseline/               # Phase 2: Baselines
├── 03_Applied_Machine_Learning_Models/    # Phase 3: Classical ML
├── 04_Deep_Learning/                      # Phase 4: Neural Networks
└── 05_Transformers/                       # Phase 5: Advanced Models
```

---

## Phase 1: Data Foundation (EDA & Preprocessing)

**Goal:** Understand the data before modeling anything  
**Location:** `01_Explanatory_Data_Analysis/`

### Tasks
-  Load & explore UCI dataset
-  Parse datetime and handle missing values (forward-fill strategy)
-  Resample from 1-minute to hourly/daily granularities
-  Exploratory Data Analysis (EDA):
  - Distribution plots (consumption across time)
  - Correlation heatmaps between features
  - Time-series plots showing daily/weekly/seasonal patterns
  - Outlier identification

### Feature Engineering
- **Temporal Features:** hour, day_of_week, month, year
- **Categorical Features:** is_weekend, season
- **Lag Features:** previous 1-hour, 24-hour consumption
- **Rolling Aggregates:** 7-day/30-day moving averages

### Outputs
- Cleaned dataset (hourly & daily resampling)
- Visualization plots (distributions, heatmaps, time series)
- Feature-engineered datasets ready for modeling

---

## Phase 2: Statistical Baselines

**Goal:** Establish benchmark models to beat  
**Location:** `02_Statistical_Baseline/`

### Baseline Models
1. **Naive Forecasts**
   - Last-value persistence (tomorrow = today's same hour)
   - Seasonal naive (same hour last week)

2. **Smoothing Methods**
   - Simple Moving Average (SMA)
   - Exponential Smoothing (ETS)
   - Holt-Winters decomposition

3. **Statistical Regression**
   - Linear Regression on time features
   - Polynomial regression for trend capture

4. **Time-Series Models**
   - SARIMA (Seasonal ARIMA) for autocorrelation patterns
   - ACF/PACF analysis for parameter tuning

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy

### Benchmarks Established
Baselines serve as reference points for Phases 3-5 model performance.

---

## Phase 3: Classical Machine Learning

**Goal:** Apply tree-based & ensemble methods  
**Location:** `03_Applied_Machine_Learning_Models/`

### Regression Models (Load Forecasting)
- **Decision Tree Regressor** — Interpretable, visual decision rules
- **Random Forest** — Bagging ensemble, feature importance scores
- **Gradient Boosting** — XGBoost, LightGBM for high accuracy
- **Support Vector Regression (SVR)** — Non-linear mapping

### Classification Tasks
1. **Peak/Off-Peak Classification** — High vs. normal consumption hours
2. **Demand Response Signals** — Predict when usage will exceed threshold
3. **Appliance Usage Detection** — Sub-metering classification

### Anomaly Detection
- Isolation Forest for outlier flagging
- Local Outlier Factor (LOF)
- Rule-based detection (sudden spikes > 2σ)

### Feature Importance
- SHAP values for model interpretability
- Permutation importance rankings

### Outputs
- Trained model artifacts (.pkl, .joblib)
- Feature importance plots
- Performance comparison tables

---

## Phase 4: Deep Learning (Temporal Sequences)

**Goal:** Capture temporal dependencies & pattern memory  
**Location:** `04_Deep_Learning/`

### Neural Network Architectures

1. **MLP (Multi-Layer Perceptron)**
   - Dense layers on engineered time features
   - Fast baseline for deep learning
   - Good for feature-rich tabular data

2. **LSTM (Long Short-Term Memory)**
   - Captures long-range dependencies
   - Ideal for sequence forecasting
   - Handles vanishing gradient problem

3. **GRU (Gated Recurrent Unit)**
   - Lighter alternative to LSTM (fewer parameters)
   - Comparable performance with faster training
   - Good for resource-constrained environments

4. **CNN-LSTM Hybrid**
   - CNN extracts local temporal patterns
   - LSTM learns long-range trends
   - Combines feature detection with sequence memory

### Tasks
- Multi-step ahead forecasting (predict next 24 hours)
- Sequence-to-sequence energy disaggregation
- Unsupervised anomaly detection (reconstruction error)

### Training Strategy
- Data normalization (MinMaxScaler)
- Train/validation/test split (70/15/15)
- Early stopping & learning rate scheduling
- Batch processing with sequences of 24-168 timesteps

### Outputs
- Trained neural network weights (.h5, .pt, .keras)
- Training/validation loss curves
- Forecast visualizations with confidence intervals

---

## Phase 5: Advanced & State-of-the-Art Models

**Goal:** Deploy cutting-edge forecasting & optimization approaches  
**Location:** `05_Transformers/`

### Advanced Architectures

1. **Transformer / Temporal Fusion Transformer (TFT)**
   - Self-attention mechanisms for long-range dependencies
   - Multi-horizon forecasting (predict multiple future timestamps)
   - Interpretable attention weights show which past events matter
   - State-of-the-art for time-series benchmarks

2. **Autoencoder for Anomaly Detection**
   - Unsupervised learning on normal consumption patterns
   - Flags deviations as anomalies
   - Useful for detecting equipment failures

3. **Sequence-to-Sequence (Seq2Seq) for NILM**
   - **NILM:** Non-Intrusive Load Monitoring (disaggregate total into appliances)
   - Predict Sub_metering_1, 2, 3 from Global_active_power
   - Encoder-Decoder architecture with attention

4. **Reinforcement Learning (Optional)**
   - **Demand-Response Optimization:** Learn optimal scheduling policies
   - State: Current consumption, time, weather
   - Action: Shift loads to off-peak hours
   - Reward: Minimized total cost

### Multi-Task Learning
- Joint forecasting of multiple sub-metering channels
- Shared representations across related tasks

### Outputs
- Production-ready forecasting API
- Inference scripts for real-time predictions
- Performance benchmarks vs. all previous phases

---

## Installation & Setup

### Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, statsmodels, TensorFlow/PyTorch, matplotlib, seaborn

### Quick Start
```bash
# Clone or navigate to project directory
cd "Household Power Consumption"

# Install dependencies
pip install pandas numpy scikit-learn statsmodels tensorflow matplotlib seaborn

# (Optional) For GPU acceleration
pip install tensorflow-gpu torch

# Run Phase 1 (EDA)
cd "01_Explanatory_Data_Analysis/notebook"
jupyter notebook power_consumption.ipynb
```

---

## Expected Results & Benchmarks

| Phase | Model | Horizon | RMSE (kW) | MAE (kW) |
|-------|-------|---------|-----------|----------|
| **2** | SARIMA | 1-hour | ~0.45 | ~0.32 |
| **3** | XGBoost | 1-hour | ~0.38 | ~0.28 |
| **4** | LSTM | 24-hour | ~0.52 | ~0.35 |
| **5** | Transformer | 24-hour | ~0.45 | ~0.32 |

*Note: Exact values depend on hyperparameter tuning and train/test split.*

---

## Key Insights (From EDA)

- **Daily Cycle:** Consumption peaks in morning (6-9 AM) and evening (5-9 PM)
- **Weekly Pattern:** Higher mid-week consumption; lower weekends
- **Seasonal Trend:** Higher in winter (heating) and summer (air-conditioning)
- **Sub-metering:** Kitchen (1) shows frequent spikes; Laundry (2) sporadic; AC (3) temperature-dependent
- **Correlations:** Active power highly correlated with current intensity & sub-metering channels

---

## References & Resources

- **Dataset:** [UCI ML Repository - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Time-Series Forecasting:** Makridakis et al., "Forecasting: methods and applications" (3rd ed.)
- **NILM:** Kelly & Knottenbelt, "The UK-Dale dataset"
- **Transformers:** Vaswani et al., "Attention Is All You Need" (2017)
- **TFT:** Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021)

---

## Contributing & Usage

This project is designed for:
-  **Learning:** Progressive ML techniques from baselines to transformers
-  **Prototyping:** Reusable code for time-series forecasting projects
-  **Production:** Energy forecasting APIs for smart grid applications

Feel free to adapt phases or swap datasets while maintaining the 5-phase structure!

---

## License & Attribution

Dataset sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)  
Original data collected by D. Efstathiou and C. Candanedo

---

## Quick Navigation

| Phase | Purpose |
|-------|--------|
| `01_Explanatory_Data_Analysis/` | Data loading, cleaning, EDA |
| `02_Statistical_Baseline/` | Naive, moving average, SARIMA |
| `03_Applied_Machine_Learning_Models/` | Trees, ensembles, anomaly detection |
| `04_Deep_Learning/` | MLP, LSTM, GRU, CNN-LSTM |
| `05_Transformers/` | Transformers, Seq2Seq, RL |

---

**Last Updated:** June 2026  
**Status:** Complete EDA + Baseline phases; Advanced phases in development
