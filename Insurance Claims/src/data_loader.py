import pandas as pd

def load_data(filepath):
    insurance = pd.read_csv(filepath)
    insurance.head()
    
    return insurance

insurance = load_data('../data/raw/insurance_claim.csv')