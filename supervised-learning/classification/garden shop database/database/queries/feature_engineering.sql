SELECT
    c.customer_id,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT oi.product_id) AS unique_products,
    COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_spent,
    COALESCE(AVG(oi.quantity * oi.unit_price), 0) AS average_item_value,
    MAX(o.order_date) AS last_order_date,
    MIN(o.order_date) AS first_order_date
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
LEFT JOIN order_items AS oi
    ON o.order_id = oi.order_id
GROUP BY c.customer_id;

