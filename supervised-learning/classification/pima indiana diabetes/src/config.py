from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'pima_indian_diabetes.csv'
MODEL_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'outputs' / 'plots'
EVAL_RESULT_DIR = BASE_DIR / 'outputs' / 'evaluation result'

FEARTURE_COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
TARGET_COLUMN = 'Outcome'