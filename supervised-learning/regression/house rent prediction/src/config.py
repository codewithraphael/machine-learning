from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'house_rent.csv'
MODEL_DIR = BASE_DIR / 'models'
REPORT_DIR = BASE_DIR / 'reports'

REQUIRED_COLS = [
    'Posted On',
    'BHK',
    'Rent',
    'Size',
    'Floor',
    'Area Type',
    'Area Locality',
    'City',
    'Furnishing Status',
    'Tenant Preferred',
    'Bathroom',
    'Point of Contact'
]


NUM_COL = [
    'bhk',
    'size',
    'bathroom'
]

CAT_COL = [
    'area_type',
    'area_locality',
    'city',
    'furnishing_status',
    'tenant_preferred',
    'point_of_contact'
]

TARGET_COL = ['rent']

RANDOM_STATE = 42