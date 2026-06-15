# %%
# ===== Loading Libraries =====

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# EDA helpers only: avoid reading files at import time.
from pathlib import Path


def load_default_data():
    """Attempt to load the dataset using available fallbacks.

    Returns
    -------
    pd.DataFrame
    """
    # Prefer the project's preprocessing loader if available
    try:
        from src.preprocessing import load_data as project_load
        return project_load('data/processed/clean_insurance.csv')
    except Exception:
        project_root = Path(__file__).resolve().parents[1]
        proc_path = project_root / 'data' / 'processed' / 'clean_insurance.csv'
        raw_path = project_root / 'data' / 'raw' / 'insurance_claim.csv'
        if proc_path.exists():
            return pd.read_csv(proc_path)
        if raw_path.exists():
            return pd.read_csv(raw_path)
        raise FileNotFoundError(
            f"Could not find dataset; looked for: {proc_path} and {raw_path}"
        )


def summary_stat(data):
    """Print basic dataset summary statistics."""
    shape = data.shape
    # capture info() output without printing twice
    print(f"\n ===== Shape of the dataset ===== \n {shape}")
    print("\n ===== Summary Statistics ===== \n")
    print(data.describe(include='all'))


def drop_column(data, columns=['policy_id']):
    df = data.copy()
    for col in columns:
        if col in df.columns:
            df = df.drop(col, axis=1)

    return df


def clean_max_power_torque(data, columns=['max_power', 'max_torque']):
    """Cleans max power and max torque columns by extracting numeric values."""
    data = data.copy()
    for col in columns:
        if col in data.columns:
            data[col] = (
                data[col]
                .astype(str)
                .str.extract(r'(\d+\.?\d*)', expand=False)
                .astype(float)
            )
    return data


def handle_outliers(data):
    q1 = data.select_dtypes(include=['number']).quantile(0.25)
    q3 = data.select_dtypes(include=['number']).quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


def check_outlier(data, lower_bound, upper_bound):
    for cols in data.select_dtypes(include=['number']).columns:
        mask = (data[cols] < lower_bound[cols]) | (data[cols] > upper_bound[cols])
        counts = mask.sum()
        if counts > 0:
            print(
                f' {cols} : {counts} outliers | '
                f'bounds = [{lower_bound[cols]:.3f}, {upper_bound[cols]:.3f}] | '
                f'actual min = {data[cols].min():.3f}, actual max = {data[cols].max():.3f}'
            )


def missing_values(data):
    values = data.isnull().sum().sort_values(ascending=False)
    print(f'\n ===== Missing Values ===== \n {values}')


def plot_numeric_box(data):
    figure = px.box(data.select_dtypes(include=['number']), orientation='h')
    figure.show()


def plot_claim_pie(data):
    claim_status_counts = data['claim_status'].value_counts()
    labels = claim_status_counts.index
    values = claim_status_counts.values
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title_text='Distribution of Claim Status')
    fig.update_traces(textinfo='percent+label')
    fig.show()


def plot_correlation_heatmap(data):
    numeric_cols = data.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_cols, annot=True, cmap='viridis', linewidths=0.5)
    plt.title('Numerical Features Heatmap Correlation')
    plt.xticks(rotation=45)
    plt.show()


def plot_pairplot(data):
    sns.pairplot(data.select_dtypes(include=['number']))
    plt.suptitle('Numerical Features Pair Plot')
    plt.show()