from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

file_path = DATA_DIR / "UR3_CobotOps.xlsx"

BINARY_COLUMNS = ['protective_stop', 'grip_lost']
TARGET = 'grip_lost'