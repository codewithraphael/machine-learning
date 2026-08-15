Large Scale Wave Energy Farm Dataset
====================================

**Overview:**

This repository folder contains materials for the "Large Scale Wave Energy Farm" dataset (sourced from the UCI Machine Learning Repository). The dataset contains measured and simulated time-series data related to wave energy conversion across a planned large-scale farm. It is useful for regression, time-series forecasting, resource assessment, and optimization tasks.

**Source & Citation:**

- **Source:** UCI Machine Learning Repository — Large Scale Wave Energy Farm (original dataset page on UCI).
- **When to cite:** If you use this data in a publication or project, cite the original dataset authors and the UCI repository page. If you need a formal citation format, check the UCI dataset page or the original paper linked from there.

**Contents of this folder:**

- `dataset/` — Raw and cleaned dataset files (CSV, Parquet or other formats). Replace with the exact filenames present in your copy of the dataset.
- `notebook/` — Jupyter notebooks with exploratory analysis and baseline models.
- `models/` — Saved model artifacts (trained model weights, evaluation summaries).
- `plots/` — Visualizations produced during analysis.

If any of the above folders are missing locally, add them or update this README to match your local layout.

**Dataset description (high level):**

- Time-indexed observations across one or more measurement points (buoys, converters, or farm-wide aggregates).
- Typical variables include (but are not guaranteed to be limited to): wave height (Hs), peak period (Tp), mean period (Tm), wave direction, significant wave steepness, power output (per device and/or farm), environmental covariates (wind speed, tide), and device/array configuration parameters.
- The dataset may contain both measured and simulated data, depending on the UCI entry and any pre-processing that was applied.

Note: column names and exact units vary by release. Inspect the files in `dataset/` to confirm the schema.

**Suggested usage examples**

Python (pandas) quick-start:

```
import pandas as pd

# adjust the filename to match your local copy
df = pd.read_csv('dataset/large_scale_wave_energy_farm.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp')
print(df.head())
```

Simple forecasting workflow:

1. Exploratory data analysis (seasonality, missingness, stationarity).
2. Feature engineering (lag features, rolling statistics, spectral features from wave time-series).
3. Baseline models: persistence, linear regression, random forest, gradient boosting, simple LSTM.
4. Evaluate with time-series-aware splits (walk-forward / rolling) and metrics such as RMSE, MAE, MAPE.

**Data cleaning notes (recommendations):**

- Check and handle missing timestamps and values; interpolation or forward-fill/backfill may be appropriate for short gaps.
- Align sensor/device timestamps to a common index if combining multiple files.
- Convert units where necessary and document any transformations.

**License & Terms of Use:**

The dataset is distributed via the UCI Machine Learning Repository. Respect the license and citation terms provided on the UCI dataset page and any original publications. If you plan to publish results, confirm license compatibility for your intended use.

**References & further reading:**

- UCI Machine Learning Repository dataset page (search for "Large Scale Wave Energy Farm").
- Any original papers or technical reports linked from the UCI entry; these usually describe how the data were collected or simulated and include metadata.

**Contact / Notes:**

---

Last updated: 2026-08-15
