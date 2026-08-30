# AirPassengers Time Series Forecasting

A complete time series forecasting project built around the classic AirPassengers dataset. This project demonstrates how to analyze seasonal demand patterns, identify trend and seasonality, and build predictive models for future monthly airline passenger counts.

## Overview

The AirPassengers dataset contains monthly airline passenger totals from 1949 to 1960. It is a classic benchmark dataset in time series analysis because it exhibits:

- a strong upward trend over time
- recurring annual seasonality
- fluctuations that require careful modeling rather than simple linear assumptions

This project focuses on understanding the data deeply and forecasting future passenger traffic using standard time series techniques.

## Objective

The main goal of this project is to:

- explore the structure and behavior of airline passenger demand over time
- detect trend and seasonality patterns
- evaluate forecasting methods
- generate future predictions and visualize the forecast
- document the modeling workflow in a reproducible and understandable format

## Dataset

- Name: AirPassengers
- Type: Monthly time series
- Time period: 1949 to 1960
- Frequency: Monthly
- Source file: [data/airline-passengers.csv](data/airline-passengers.csv)
- Features:
  - Month
  - Passengers

The dataset is widely used for time series forecasting experiments and is well suited for modeling seasonal effects and long-term growth.

## Project Structure

```text
AirPassengers/
├── data/
│   └── airline-passengers.csv
├── notebook/
│   └── eda.ipynb
├── plots/
├── models/
├── evaluation result/
├── src/
│   ├── analysis.py
│   ├── data_loader.py
│   ├── forecasting.py
│   ├── preprocessing.py
│   └── evaluation.py
├── README.md
└── requirements.txt (if added in the project folder)
```

## Workflow

The workflow follows a standard time series analysis pipeline:

1. Data loading and inspection
2. Exploratory data analysis (EDA)
3. Time series decomposition
4. Trend and seasonality analysis
5. Stationarity checks
6. Model selection and training
7. Forecast evaluation
8. Visualization of results and future predictions

## Methodology

This project typically applies a combination of the following approaches:

- descriptive analysis of the time series
- decomposition into trend, seasonality, and residual components
- visualization of monthly patterns and long-term movement
- modeling using classical forecasting methods such as:
  - moving average
  - exponential smoothing
  - ARIMA / SARIMA
  - seasonal forecasting techniques
- evaluation using metrics such as MAE, RMSE, and MAPE

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Statsmodels
- Jupyter Notebook

## Setup

To run the project locally:

```bash
git clone https://github.com/codewithraphael/machine-learning.git
cd machine-learning/time-series-analysis/AirPassengers
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install pandas numpy matplotlib seaborn statsmodels jupyter
```

If the repository-level requirements file is used, you can also install dependencies with:

```bash
pip install -r ../../requirements.txt
```

## Usage

1. Open the notebook in the [notebook](notebook) folder to view the exploratory analysis.
2. Review the time series plots, seasonal patterns, and decomposition results.
3. Use the scripts in [src](src) for preprocessing, analysis, and forecasting tasks.
4. Save generated figures in the [plots](plots) folder.
5. Store trained models and evaluation outputs in [models](models) and [evaluation result](evaluation%20result).

## Expected Insights

From the AirPassengers dataset, we typically expect to observe:

- a clear long-term upward trend in passenger traffic
- strong yearly seasonal cycles
- higher travel demand during certain months of the year
- the need for seasonal-aware forecasting models to capture the repeating pattern

## Evaluation

Model performance is evaluated using standard forecasting accuracy metrics and visual comparison between actual and predicted values. Forecast results are stored in the evaluation output directory for review and comparison.

## Results

The project generates:

- time series plots
- seasonal decomposition plots
- forecast visualizations
- model evaluation summaries
- final predictions for future periods

These outputs help assess how well the chosen model captures both trend and seasonality.

## Conclusion

This project provides a practical example of time series forecasting using a well-known real-world dataset. It is useful for learning the fundamentals of time series analysis, forecasting methodology, and how to evaluate model performance in a seasonal setting.

## License

This project is part of the broader machine learning repository and is intended for educational and research purposes.
