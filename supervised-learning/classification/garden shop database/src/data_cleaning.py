import pandas as pd


def clean_query(query):

    '''
    dropping columns with None or Zero values,
    converting first_order_date and last_order_date to Datetime
    Index by last_order_date
    '''

    query = query.dropna()
    
    date_columns = ['first_order_date', 'last_order_date']
    query[date_columns] = query[date_columns].apply(pd.to_datetime)

    query.set_index('last_order_date', inplace=True)



    print('='*100)
    print(' '*40 + 'CLEANED GARDEN SHOP DATABASE')
    print('='*100)

    print(query)
    
    return query