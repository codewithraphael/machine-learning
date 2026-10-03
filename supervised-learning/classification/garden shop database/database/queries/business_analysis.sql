SELECT
    p.product_name,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM order_items AS oi
JOIN products AS p
    ON oi.product_id = p.product_id
GROUP BY p.product_name
ORDER BY revenue DESC;


SELECT
    cat.category_name,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM order_items AS oi
JOIN products AS p
    ON oi.product_id = p.product_id
JOIN categories AS cat
    ON p.category_id = cat.category_id
GROUP BY cat.category_name
ORDER BY revenue DESC;


SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    SUM(oi.quantity * oi.unit_price) AS total_spent
FROM customers AS c
JOIN orders AS o
    ON c.customer_id = o.customer_id
JOIN order_items AS oi
    ON o.order_id = oi.order_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spent DESC;