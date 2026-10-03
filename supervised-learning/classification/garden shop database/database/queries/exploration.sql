SELECT COUNT(*) AS customers FROM customers;
SELECT COUNT(*) AS products FROM products;
SELECT COUNT(*) AS categories FROM categories;
SELECT COUNT(*) AS orders FROM orders;
SELECT COUNT(*) AS order_items FROM order_items;
SELECT COUNT(*) AS suppliers FROM suppliers;

SELECT * FROM customers LIMIT 10;
SELECT * FROM products LIMIT 10;
SELECT * FROM categories LIMIT 10;
SELECT * FROM orders LIMIT 10;
SELECT * FROM order_items LIMIT 10;
SELECT * FROM suppliers LIMIT 10;



SELECT
    o.order_id,
    c.customer_id,
    c.first_name,
    c.last_name,
    o.order_date,
    o.status
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
ORDER BY o.order_date;



SELECT
    o.order_id,
    p.product_id,
    oi.quantity,
    oi.unit_price
FROM order_items oi
JOIN orders o 
    ON o.order_id = oi.order_id
JOIN products p 
    ON oi.product_id = p.product_id
ORDER BY o.order_date, p.product_name;