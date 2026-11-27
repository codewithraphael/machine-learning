import pandas as pd


def load_data(filepath: pd.DataFrame) -> pd.DataFrame:

    try:
        if not filepath.exists():
            raise FileExistsError(f'file does not exist in path')

        else:
            data = pd.read_csv(filepath)

            return data
    except Exception as e:
        print(f'failed to load data {e}')