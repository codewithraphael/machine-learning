# Rainfall Prediction Dataset

This dataset contains daily weather observations used to predict rainfall. Each row represents one day, with features describing atmospheric and environmental conditions. The target is Rainfall, which can be used for either classification (rain or no rain) or regression (amount of rain).

## Features:

1. Day: Date or time of observation
2. Pressure: Atmospheric pressure
3. MaxTemp: Maximum temperature of the day
4. Temperature: Average daily temperature
5. MinTemp: Minimum temperature of the day
6. DewPoint: Moisture level in the air
7. Humidity: Relative humidity percentage
8. Cloud: Cloud cover level
9. Rainfall: Target variable (rain amount or yes or no)
10. Sunshine: Hours of sunlight
11. WindDirection: Direction of wind
12. WindSpeed: Speed of wind

## Problem Type:

1. Classification: Predict if it will rain (0 or 1)
2. Regression: Predict how much rain will fall

## Preprocessing:

1. Handle missing values using mean, median, or mode
2. Scale features if using linear models
3. Tree based models do not require scaling

## Model Evaluation:

1. Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
2. Regression: MAE, MSE, RMSE, R² Score