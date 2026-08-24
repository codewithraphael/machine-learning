from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

file_path = DATA_DIR / "UR3_CobotOps.xlsx"

BINARY_COLUMNS = ['protective_stop', 'grip_lost']
TARGET = 'grip_lost'
HORIZON = 1
RANDOM_STATE = 42

SENSOR_FEATURES = [
    'current_j0',
    'temperature_t0',
    'current_j1',
    'temperature_j1',
    'current_j2',
    'temperature_j2',
    'current_j3',
    'temperature_j3',
    'current_j4',
    'temperature_j4',
    'current_j5',
    'temperature_j5',
    'speed_j0',
    'speed_j1',
    'speed_j2',
    'speed_j3',
    'speed_j4',
    'speed_j5',
    'tool_current'
]
FUTURE_TARGET = f'{TARGET}_t_plus_{HORIZON}'
TRAIN_RATIO = 0.8 