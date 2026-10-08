import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_QUERY = '''

SELECT
    c.customer_id,
    c.city,
    c.state,
    c.is_active,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT oi.product_id) AS unique_products,
    COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_spent,
    COALESCE(AVG(oi.quantity * oi.unit_price), 0) AS average_item_value,
    MAX(o.order_date) AS last_order_date,
    MIN(o.order_date) AS first_order_date
FROM customers AS c
LEFT JOIN orders as o
    ON c.customer_id = o.customer_id
LEFT JOIN order_items as oi
    ON o.order_id = oi.order_id
GROUP BY
    c.customer_id,
    c.city,
    c.state,
    c.is_active

'''

TARGET_QUERY = '''

SELECT
    c.customer_id,
    CASE WHEN COUNT(o.order_id) = 0 THEN 1 ELSE 0 END AS churned
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
    AND o.order_date >= :cutoff
    AND o.order_date < date(:cutoff, '+90 days')
GROUP BY c.c.customer_id

'''

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'reports' / 'plots'

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'garden_shop')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD  = os.getenv('DB_PASSWORD')

NUM_COLUMNS = [
    'total_orders'
]