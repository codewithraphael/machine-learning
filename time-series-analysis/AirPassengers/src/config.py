from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / 'data' / 'airline-passengers.csv'
PLOTS_PATH = ROOT / 'plots'
MODELS_PATH = ROOT / 'models'

filepath = DATA_PATH