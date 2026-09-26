from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cervical_cancer_data.csv"
MODEL_PATH = BASE_DIR / "models"
PLOTS_PATH = BASE_DIR / "plots"

TARGET_COLUMN = 'ca_cervix'

FEATURE_COLUMNS = [
    'behavior_sexualrisk',
    'behavior_eating',
    'behavior_personalhygine',
    'intention_aggregation',
    'intention_commitment',
    'attitude_consistency',
    'attitude_spontaneity',
    'norm_significantperson',
    'norm_fulfillment',
    'perception_vulnerability',
    'perception_severity',
    'motivation_strength',
    'motivation_willingness',
    'socialsupport_emotionality',
    'socialsupport_appreciation',
    'socialsupport_instrumental',
    'empowerment_knowledge',
    'empowerment_abilities',
    'empowerment_desires'
]