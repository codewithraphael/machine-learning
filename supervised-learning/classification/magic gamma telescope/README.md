# MAGIC Gamma Telescope Dataset

Binary classification project based on the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) from the UCI Machine Learning Repository.

## Overview

The dataset contains simulated measurements of atmospheric Cherenkov shower images recorded by a ground-based gamma-ray telescope. The objective is to distinguish:

- **Gamma (`g`)**: a gamma-ray event (signal)
- **Hadron (`h`)**: a hadronic cosmic-ray event (background)

The measurements are derived from the shape and intensity of shower images and include several Hillas parameters commonly used for event discrimination.

## Dataset Summary

| Property | Value |
| --- | --- |
| Instances | 19,020 |
| Predictors | 10 continuous features |
| Target | `class` |
| Classes | `g` (gamma), `h` (hadron) |
| Missing values | None reported |
| File format | Comma-separated values without a header |

### Class Distribution

| Class | Meaning | Instances | Proportion |
| --- | --- | ---: | ---: |
| `g` | Gamma signal | 12,332 | 64.8% |
| `h` | Hadron background | 6,688 | 35.2% |

The original documentation notes that the hadron class is underestimated for technical reasons. Consequently, accuracy alone is not a sufficient measure of model quality.

## Features

| Feature | Description | Unit or scale |
| --- | --- | --- |
| `fLength` | Major axis of the shower-image ellipse | millimetres |
| `fWidth` | Minor axis of the shower-image ellipse | millimetres |
| `fSize` | Base-10 logarithm of the total pixel content | log10 photons |
| `fConc` | Ratio of the two highest pixel values to `fSize` | ratio |
| `fConc1` | Ratio of the highest pixel value to `fSize` | ratio |
| `fAsym` | Distance from the highest pixel to the image centre, projected onto the major axis | millimetres |
| `fM3Long` | Cube root of the third moment along the major axis | millimetres |
| `fM3Trans` | Cube root of the third moment along the minor axis | millimetres |
| `fAlpha` | Angle between the major axis and the vector to the origin | degrees |
| `fDist` | Distance from the origin to the ellipse centre | millimetres |
| `class` | Event label: gamma or hadron | categorical |

## Project Structure

```text
magic gamma telescope/
├── api/                 # API entry point
├── data/
│   ├── magic04.data     # Raw observations
│   └── magic04.names    # Original dataset documentation
├── notebook/
│   └── eda.ipynb        # Exploratory data analysis
├── src/                 # Data preparation, evaluation, and visualization modules
└── README.md
```

## Getting Started

### 1. Load the data

Because `magic04.data` has no header row, provide the column names explicitly:

```python
import pandas as pd

columns = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class",
]

data = pd.read_csv(
    "data/magic04.data",
    header=None,
    names=columns,
)
```

### 2. Prepare the target

For scikit-learn models, encode the target as a binary value or use a classifier that accepts string labels:

```python
X = data.drop(columns="class")
y = data["class"].map({"h": 0, "g": 1})
```

### 3. Evaluate responsibly

Use a stratified train/test split and report metrics that reflect the cost of false positives and false negatives. Recommended metrics include:

- ROC curve and ROC-AUC
- Precision-recall curve and average precision
- Confusion matrix
- Signal efficiency and background acceptance at an explicitly stated threshold

The original UCI documentation specifically recommends comparing classifiers with ROC curves and examining low background-acceptance operating points, such as 0.01, 0.02, 0.05, 0.10, and 0.20.

## Recommended Workflow

1. Load the headerless data and assign the documented column names.
2. Verify data types, class labels, duplicates, and missing values.
3. Inspect feature distributions and potential outliers.
4. Split the data with stratification to preserve class proportions.
5. Scale features when required by the selected algorithm.
6. Train one or more binary classification models.
7. Compare models using threshold-aware metrics and ROC analysis.
8. Select an operating threshold based on the intended scientific or business objective.

## Data Provenance

The dataset was developed by the [Major Atmospheric Gamma Imaging Cherenkov Telescope (MAGIC) project](http://wwwmagic.mppmu.mpg.de/) and donated to the UCI Machine Learning Repository by P. Savicky. The data were received by UCI in May 2007 and were generated with Monte Carlo simulations using CORSIKA.

For the complete attribute definitions, scientific background, and original references, see [`data/magic04.names`](data/magic04.names).

## References

1. Bock, R. K., et al. (2004). *Methods for multidimensional event classification: a case study using images from a Cherenkov gamma-ray telescope*. Nuclear Instruments and Methods in Physics Research Section A, 516, 511–528.
2. UCI Machine Learning Repository. [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).
3. Heck, D., et al. (1998). *CORSIKA: A Monte Carlo code to simulate extensive air showers*. Forschungszentrum Karlsruhe FZKA 6019.