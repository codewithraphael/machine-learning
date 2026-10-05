import pandas as pd

from pathlib import Path
from database import get_connection
from config import FEATURE_QUERY


def load_sql_query():

    connection = get_connection()

    try:
        query = pd.read_sql_query(FEATURE_QUERY, connection)
    finally:
        connection.close()

    return query