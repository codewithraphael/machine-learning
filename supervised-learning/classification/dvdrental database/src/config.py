from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'customer_rental_prediction.csv'
MODEL_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'outputs' / 'plots'
EVAL_RESULT_DIR = BASE_DIR / 'outputs' / 'evaluation result'