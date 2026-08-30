from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / 'data / airline-passengers.csv'
PLOTS_PATH = ROOT / 'plots'
MODELS_PATH = ROOT / 'models'

DATA_PATH.mkdir(exist_ok=True, parents=True)
PLOTS_PATH.mkdir(exist_ok=True, parents=True)
MODELS_PATH.mkdir(exist_ok=True, parents=True)

filepath = DATA_PATH