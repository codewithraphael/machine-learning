import pandas as pd

def load_data(filepath):
    """Load a CSV file and return a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    data = pd.read_csv(filepath)
    return data