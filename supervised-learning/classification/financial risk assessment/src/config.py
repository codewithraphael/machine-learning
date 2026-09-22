from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data/financial_risk_assessment.csv'
MODELS_PATH = PROJECT_ROOT / 'models'
PLOTS_PATH = PROJECT_ROOT / 'plots'
RESULTS_PATH = PROJECT_ROOT / 'evaluation_result'

CAT_COLUMNS = [
    'gender',
    'education_level',
    'marital_status',
    'loan_purpose',
    'employment_status',
    'payment_history',
    'city',
    'state',
    'country'
]

NUM_COLUMNS = [
    'age',
    'income',
    'credit_score',
    'loan_amount',
    'years_at_current_job',
    'debt_to_income_ratio',
    'assets_value',
    'number_of_dependents',
    'previous_defaults',
    'marital_status_change'
]