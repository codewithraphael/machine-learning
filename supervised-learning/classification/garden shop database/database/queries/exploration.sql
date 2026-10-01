SELECT COUNT(*) AS customers FROM customers;
select COUNT(*) AS products FROM products;
select COUNT(*) AS categories FROM categories;
select COUNT(*) AS orders FROM orders;
select COUNT(*) AS order_items FROM order_items;
select COUNT(*) AS suppliers FROM suppliers;

SELECT * FROM customers LIMIT 10;
SELECT * FROM products LIMIT 10;
select * FROM categories LIMIT 10;
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
ORDER BY o.order_date DESC;