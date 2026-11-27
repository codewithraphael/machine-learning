import pandas as pd
import re

from config import REQUIRED_COLS


_ORDINALS_BELOW_TWENTY = {
    1: 'first',
    2: 'second',
    3: 'third',
    4: 'fourth',
    5: 'fifth',
    6: 'sixth',
    7: 'seventh',
    8: 'eighth',
    9: 'ninth',
    10: 'tenth',
    11: 'eleventh',
    12: 'twelfth',
    13: 'thirteenth',
    14: 'fourteenth',
    15: 'fifteenth',
    16: 'sixteenth',
    17: 'seventeenth',
    18: 'eighteenth',
    19: 'nineteenth',
}

_TENS = {
    20: 'twenty',
    30: 'thirty',
    40: 'forty',
    50: 'fifty',
    60: 'sixty',
    70: 'seventy',
    80: 'eighty',
    90: 'ninety',
}

_ORDINAL_TENS = {
    20: 'twentieth',
    30: 'thirtieth',
    40: 'fortieth',
    50: 'fiftieth',
    60: 'sixtieth',
    70: 'seventieth',
    80: 'eightieth',
    90: 'ninetieth',
}


def _floor_to_label(value: object) -> str:
    floor = str(value).strip().casefold()

    special_floors = {
        'ground': 'ground floor',
        'lower basement': 'lower basement',
        'upper basement': 'upper basement',
    }
    special_match = re.fullmatch(
        r'(ground|lower basement|upper basement)(?:\s+out\s+of\s+\d+)?',
        floor,
    )
    if special_match:
        return special_floors[special_match.group(1)]

    numeric_match = re.fullmatch(r'(\d+)(?:\s+out\s+of\s+\d+)?', floor)
    if numeric_match:
        number = int(numeric_match.group(1))
        if number in _ORDINALS_BELOW_TWENTY:
            ordinal = _ORDINALS_BELOW_TWENTY[number]
        elif number in _ORDINAL_TENS:
            ordinal = _ORDINAL_TENS[number]
        elif 20 < number < 100:
            tens, ones = divmod(number, 10)
            ordinal = f'{_TENS[tens * 10]}-{_ORDINALS_BELOW_TWENTY[ones]}'
        else:
            raise ValueError(f'unsupported floor number: {number}')
        return f'{ordinal} floor'

    raise ValueError(f'unrecognized floor value: {value!r}')


def validate_data(data: pd.DataFrame) -> None:

    '''
    data validation

    checks:
        - dataset is not empty
        - missing values in dataset columns
    
    args:
        data (pd. DataFrame)

    returns:
        None
    
    '''

    if data.empty:
        raise ValueError(f'dataset is empty')

    for col in REQUIRED_COLS:
        if data[col].isnull().any():
            raise ValueError(f'missing data found in {col}')

        if data[col].astype(str).str.strip().eq(' ').any():
            raise ValueError(f'blank spaces found in {col}')


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(
        columns={
            col: col.lower().replace(' ', '_')
            for col in REQUIRED_COLS
            if col in data
        }
    )

    if 'floor' in data:
        data['floor'] = data['floor'].map(_floor_to_label)

    return data