from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data' / 'magic04.data'
NAMES = BASE_DIR / 'data' / 'magic04.names'
MODELS_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'outputs' / 'plots'
EVAL_RESULT_DIR = BASE_DIR / 'outputs' / 'evaluation result'

COL_NAMES = [
    'fLength', 
    'fWidth', 
    'fSize', 
    'fConc', 
    'fConc1', 
    'fAsym', 
    'fM3Long', 
    'fM3Trans', 
    'fAlpha', 
    'fDist', 
    'class'
]

FEATURE_COLS = [
    'fLength', 
    'fWidth', 
    'fSize', 
    'fConc', 
    'fConc1', 
    'fAsym', 
    'fM3Long', 
    'fM3Trans', 
    'fAlpha', 
    'fDist'
]

TARGET_COL = 'class'

random_state = 42