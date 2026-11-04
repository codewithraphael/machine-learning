import pandas as pd

def load_data(filepath):

    """
    Load the dataset from the specified filepath.

    Args:
        filepath (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.
    """


    try:
        data = pd.read_csv(filepath)
        
        return data
    
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")