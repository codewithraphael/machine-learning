import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SQL_FILE = BASE_DIR / 'database / queries / ml_dataset.sql'
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'reports / plots'

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'garden_shop')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD  = os.getenv('DB_PASSWORD')