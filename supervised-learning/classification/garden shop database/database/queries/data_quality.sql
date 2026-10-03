-- missing values --

SELECT
    COUNT(*) AS total_rows,
    COUNT(phone) AS non_null_phone ,
    COUNT (*) - COUNT(phone) AS missing_phone_numbers
FROM customers;

SELECT
    COUNT(*) AS total_orders,
    COUNT(order_date) AS non_null_order_dates,
    COUNT(*) - COUNT(order_date) AS missing_order_dates
FROM orders;

-- duplicate IDs --

SELECT
    customer_id,
    COUNT(*)
FROM customers
GROUP BY customer_id
HAVING COUNT(*) > 1;

SELECT
    product_id,
    COUNT(*)
FROM  products
GROUP BY product_id
HAVING COUNT(*) > 1;

-- invalid values --

SELECT *
FROM products
WHERE price < 0;

SELECT *
FROM order_items
WHERE quantity <= 0
    OR unit_price < 0;

-- missing dates --

SELECT *
from orders
WHERE order_date is NULL;

