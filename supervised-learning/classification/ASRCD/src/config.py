from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'ASRCD_dataset.csv'
MODEL_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'reports' / 'plots'
EVAL_RESULT_DIR = BASE_DIR / 'report' / 'results'

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_FEATURES = 30000
MAX_ITER = 2000

REQUIRED_COLUMNS = {
    'id',
    'query',
    'response_class',
    'harm_category',
    'risk_level',
    'explanation'
}

TARGET_COLUMN = ['response_class', 'harm_category', 'risk_level']
TEXT_COLUMN = 'query'
