import pandas as pd

def features(query):

    query['average_order_value'] = (
        query['total_spent'] / query['total_orders'].replace(0, 1)
    )

    query['products_per_order'] = (
        query['unique_products'] / query['total_orders']
    ).round().astype(int)

    print('='*100)
    print(' '*40 + 'FEATURE ENGINEERING')
    print('='*100)

    print(query)

    return query