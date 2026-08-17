# UR3 CobotOps Dataset — Project README

## 1. Overview

The **UR3 CobotOps** dataset is a multivariate time-series dataset collected from a **Universal Robots UR3** collaborative robot ("cobot") during repeated pick-and-place operations. It records the robot's internal operating state — joint currents, temperatures, and speeds, gripper (tool) current, cycle counts — alongside two fault indicators: **protective stops** and **grip losses**.

It was donated to the UCI Machine Learning Repository on **28 February 2024** by researchers from the University of Ioannina, Politecnico di Torino, and the Industrial Systems Institute (Athena RC).

| Property | Value |
|---|---|
| Repository | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/963/ur3+cobotops) |
| Dataset ID | 963 |
| DOI | [10.24432/C5J891](https://doi.org/10.24432/C5J891) |
| Domain | Engineering / Industrial Robotics |
| Data type | Multivariate, Time-Series |
| Instances | 7,409 (some sources cite 7,410) |
| Features | 20 |
| Targets | 2 (`Grip_lost`, `Protective_stop`) |
| Missing values | Yes |
| License | CC BY 4.0 |
| Source file | `dataset_02052023.xlsx` (~1.7 MB) |

## 2. Background

The UR3 is a compact, lightweight collaborative robot arm built for automating repetitive tasks (pick-and-place, assembly, light material handling) in shared human-robot workspaces. Data was collected via the **MODBUS** and **RTDE (Real-Time Data Exchange)** protocols while the robot performed a controlled pick-and-place task between two fixed waypoints (a pickup point and a drop point), with varying speed, workload, and gripping force to induce operational stress and, occasionally, faults.

The dataset was created to support research into **fault diagnosis, predictive maintenance, and operational optimization** for collaborative robots in industrial settings.

## 3. What Counts as an "Instance"

Each **instance (row)** in the dataset is a **single timestamped snapshot of the robot's operating state**, sampled at a point during a pick-and-place cycle. Every row captures:

- The instantaneous electrical current and temperature of each of the robot's 6 joints
- The instantaneous speed of each joint
- The gripper's (tool's) current draw
- Which operation cycle the sample belongs to
- Whether a **protective stop** or **grip loss** fault occurred at/around that sample

In other words, the dataset is not one row per completed job — it's a **fine-grained sensor log across the robot's 6 joints**, recorded repeatedly throughout many operation cycles, with fault events labeled where they occurred. Because faults are rare relative to normal operation, the dataset is **heavily class-imbalanced**: one downstream study reports roughly **7,077 non-fault samples vs. 278 protective-stop events** out of ~7,410 rows (~3.8% positive class).

## 4. Robot Structure Context

The UR3 has **6 degrees of freedom**, i.e., 6 joints, commonly indexed **J0–J5**:

| Joint | Common name |
|---|---|
| J0 | Base |
| J1 | Shoulder |
| J2 | Elbow |
| J3 | Wrist 1 |
| J4 | Wrist 2 |
| J5 | Wrist 3 |

Every current/temperature/speed feature below is duplicated across these 6 joints.

## 5. Features (20)

| Feature | Type | Unit / Notes |
|---|---|---|
| `Current_J0` – `Current_J5` | Continuous | Electrical current drawn by each joint's motor (Amps). Reflects torque/load on that joint. |
| `Temperature_T0` / `Temperature_J1`–`Temperature_J5` | Continuous | Internal motor/joint temperature (°C) for each joint. |
| `Speed_J0` – `Speed_J5` | Continuous | Angular velocity of each joint. |
| `Tool_current` | Continuous | Current drawn by the gripper/end-effector (Amps). Reflects gripping force / load on the tool. |
| `cycle` | Integer | Identifier/count of the pick-and-place operation cycle the sample belongs to. |

*(6 currents + 6 temperatures + 6 speeds + tool current + cycle = 20 features.)*

## 6. Targets (2)

| Target | Type | Meaning |
|---|---|---|
| `Protective_stop` | Boolean / Categorical (`True`/`False`) | Whether the robot's built-in safety system halted operation at that point — typically triggered by abnormal joint current/force, unexpected resistance, or excessive speed changes, to protect the robot, workpiece, or nearby humans. |
| `Grip_lost` | Boolean / Categorical (`True`/`False`) | Whether the gripper failed to hold or dropped the workpiece during the cycle. |

These are the two fault modes the dataset is designed to help predict. Some works treat them as two separate binary classification problems; others combine them into a single multi-class or multi-label fault target.

## 7. Associated Machine Learning Tasks

The dataset is explicitly tagged for multiple ML task types:

1. **Classification (primary use case)** — Predict `Protective_stop` and/or `Grip_lost` from the sensor readings. This is the main studied task (fault detection / predictive maintenance). Because faults are rare, this is effectively an **imbalanced binary classification** problem, and papers using it report low recall (best reported ~34.7% with KNN in one baseline study) unless imbalance-handling techniques are used.
2. **Regression** — Predicting continuous operational parameters (e.g., forecasting current/temperature/speed trends) rather than discrete fault labels.
3. **Clustering** — Unsupervised discovery of operational regimes or anomalous behavior patterns without relying on the fault labels.
4. **Other (causal/interpretable modeling)** — The dataset's introductory paper uses it with **Fuzzy Cognitive Maps (FCMs)** to model interpretable causal relationships between sensor variables and fault occurrence (e.g., identifying that high workload and abnormal `Current_J1`/`Current_J2` are strong drivers of protective stops).

### Known findings from prior research
- ANOVA analysis found **workload** to be the most significant factor for protective stops (F = 7.78, p = 0.0032).
- `Current_J1` and `Current_J2` were consistently identified as top predictors of protective stops.
- `Current_J2`, `Speed_J4`, and `Speed_J5` were identified as key predictors of grip-loss faults.
- Feature selection (RFE, Chi-square) narrowing to the top 10 features has been shown to improve Decision Tree and Random Forest classifier performance on this dataset.

## 8. Data Quality Notes

- **Missing values are present** — expect to handle NaNs before modeling (imputation or row/column filtering).
- **Severe class imbalance** in both fault targets — plan for techniques such as class weighting, SMOTE/undersampling, stratified splits, or anomaly-detection framing rather than naive accuracy-based evaluation.
- Data is **time-ordered within cycles** — a random train/test split can leak information across a cycle; consider cycle-aware or time-based splitting for realistic evaluation.

## 9. Getting the Data

### Option A: `ucimlrepo` package
```bash
pip install ucimlrepo
```
```python
from ucimlrepo import fetch_ucirepo

# fetch dataset
ur3_cobotops = fetch_ucirepo(id=963)

# data (as pandas dataframes)
X = ur3_cobotops.data.features
y = ur3_cobotops.data.targets

# metadata
print(ur3_cobotops.metadata)

# variable information
print(ur3_cobotops.variables)
```

### Option B: Direct download
- File: `dataset_02052023.xlsx` (~1.7 MB)
- Source: https://archive.ics.uci.edu/static/public/963/ur3+cobotops.zip

## 10. Suggested Project Structure

```
ur3-cobotops/
├── data/
│   └── dataset_02052023.xlsx
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
|   |-- main.py
│   └── evaluate.py
├── models/
├── README.md
└── requirements.txt
```

## 11. Citation

If you use this dataset, cite:

```
Tyrovolas, M., Aliev, K., Antonelli, D., & Stylios, C. (2024).
UR3 CobotOps [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5J891
```

Related paper:

```
Tyrovolas, M., Stylios, C., Aliev, K., & Antonelli, D. (2024).
Leveraging Information Flow-Based Fuzzy Cognitive Maps for Interpretable
Fault Diagnosis in Industrial Robotics. 15th Advanced Doctoral Conference
on Computing, Electrical and Industrial Systems (DoCEIS).
```

## 12. License

This dataset is released under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license — free to share and adapt for any purpose with appropriate credit.

## 13. Creators / Contact

| Name | Affiliation | Email |
|---|---|---|
| Marios Tyrovolas | Dept. of Informatics & Telecommunications, University of Ioannina | tirovolas@kic.uoi.gr |
| Khurshid Aliev | Dept. of Management and Production Engineering, Politecnico di Torino | khurshid@polito.it |
| Dario Antonelli | Dept. of Management and Production Engineering, Politecnico di Torino | dario.antonelli@polito.it |
| Chrysostomos Stylios | Industrial Systems Institute (ISI), Athena RC | stylios@isi.gr |
