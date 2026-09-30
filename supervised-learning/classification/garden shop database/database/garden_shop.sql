-- Garden Shop - SQL practice dataset
-- Orders, customers, products, and inventory for a small garden supply business.
--
-- Free to use for any purpose, including commercially. No attribution required.
-- Browse, query, and re-download at https://sqlshed.com/practice-datasets/
--
-- 6 tables, 130 rows. Standard SQL: runs on PostgreSQL, MySQL, SQLite,
-- and DuckDB without edits. SQL Server has no BOOLEAN type: change
-- BOOLEAN to BIT and TRUE/FALSE to 1/0 before running it there.

DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- customers: One row per customer. Some customers have no phone on file.
CREATE TABLE customers (
  customer_id INTEGER,
  first_name VARCHAR(255),
  last_name VARCHAR(255),
  email VARCHAR(255),
  phone VARCHAR(255),
  city VARCHAR(255),
  state VARCHAR(255),
  signup_date DATE,
  is_active BOOLEAN
);

INSERT INTO customers (customer_id, first_name, last_name, email, phone, city, state, signup_date, is_active) VALUES
  (1, 'Theo', 'Brandt', 'theo.brandt@example.com', '215-555-0142', 'Philadelphia', 'PA', '2023-02-14', TRUE),
  (2, 'Nadia', 'Reyes', 'nadia.reyes@example.com', NULL, 'Pittsburgh', 'PA', '2023-05-03', TRUE),
  (3, 'Owen', 'Castellano', 'owen.c@example.com', '267-555-0199', 'Allentown', 'PA', '2024-01-20', TRUE),
  (4, 'Mia', 'Thompson', 'mia.t@example.com', '415-555-0110', 'San Francisco', 'CA', '2022-11-30', TRUE),
  (5, 'Liam', 'Walsh', 'liam.walsh@example.com', NULL, 'Columbus', 'OH', '2023-08-17', FALSE),
  (6, 'Ava', 'Nguyen', 'ava.nguyen@example.com', '503-555-0188', 'Portland', 'OR', '2024-03-11', TRUE),
  (7, 'Noah', 'Patel', 'noah.patel@example.com', '305-555-0123', 'Miami', 'FL', '2023-06-25', TRUE),
  (8, 'Emma', 'Brandt', 'emma.brandt@example.com', '215-555-0177', 'Philadelphia', 'PA', '2024-02-02', FALSE),
  (9, 'Lucas', 'Romano', 'lucas.r@example.com', NULL, 'Cleveland', 'OH', '2022-09-09', TRUE),
  (10, 'Sofia', 'Greer', 'sofia.greer@example.com', '415-555-0166', 'Oakland', 'CA', '2023-12-19', TRUE),
  (11, 'Ethan', 'Kim', 'ethan.kim@example.com', '206-555-0144', 'Seattle', 'WA', '2024-04-05', TRUE),
  (12, 'Isabella', 'Ford', 'isabella.ford@example.com', NULL, 'Austin', 'TX', '2023-03-28', TRUE),
  (13, 'Mason', 'Doyle', 'mason.doyle@example.com', '412-555-0133', 'Pittsburgh', 'PA', '2024-05-15', TRUE),
  (14, 'Chloe', 'Bauer', 'chloe.bauer@example.com', '503-555-0121', 'Eugene', 'OR', '2022-07-22', FALSE),
  (15, 'Logan', 'Pierce', 'logan.pierce@example.com', '305-555-0150', 'Orlando', 'FL', '2023-10-30', TRUE),
  (16, 'Harper', 'Quinn', 'harper.quinn@example.com', NULL, 'Sacramento', 'CA', '2024-06-01', TRUE),
  (17, 'Jack', 'Mercer', 'jack.mercer@example.com', '614-555-0109', 'Columbus', 'OH', '2023-01-12', TRUE),
  (18, 'Lily', 'Castellano', 'lily.c@example.com', '267-555-0102', 'Allentown', 'PA', '2024-03-19', TRUE),
  (19, 'Daniel', 'Foss', 'daniel.foss@example.com', '206-555-0175', 'Tacoma', 'WA', '2022-12-05', FALSE),
  (20, 'Grace', 'Holt', 'grace.holt@example.com', '512-555-0190', 'Austin', 'TX', '2024-04-28', TRUE);

-- products: One row per product, with price, cost, and inventory levels.
CREATE TABLE products (
  product_id INTEGER,
  product_name VARCHAR(255),
  category_id INTEGER,
  supplier_id INTEGER,
  price DECIMAL(12,2),
  cost DECIMAL(12,2),
  quantity_on_hand INTEGER,
  reorder_level INTEGER,
  discontinued BOOLEAN
);

INSERT INTO products (product_id, product_name, category_id, supplier_id, price, cost, quantity_on_hand, reorder_level, discontinued) VALUES
  (1, 'Fiddle-Leaf Fig', 1, 1, 32.00, 18.00, 14, 10, FALSE),
  (2, 'Monstera Deliciosa', 1, 1, 28.00, 15.00, 40, 15, FALSE),
  (3, 'Snake Plant', 1, 4, 18.00, 9.00, 55, 20, FALSE),
  (4, 'Pothos ''Golden''', 1, 4, 12.00, 5.00, 8, 12, FALSE),
  (5, 'ZZ Plant', 1, 1, 24.00, 12.00, 30, 10, FALSE),
  (6, 'Sunflower Seeds', 2, 6, 3.50, 1.20, 200, 50, FALSE),
  (7, 'Tomato Seeds ''Roma''', 2, 6, 2.75, 0.90, 5, 40, FALSE),
  (8, 'Tulip Bulbs (10pk)', 2, 6, 8.00, 3.50, 120, 30, FALSE),
  (9, 'Pruning Shears', 3, 5, 22.00, 11.00, 18, 8, FALSE),
  (10, 'Garden Trowel', 3, 5, 9.50, 4.00, 60, 15, FALSE),
  (11, 'Hand Cultivator', 3, 5, 8.00, 3.50, 3, 10, FALSE),
  (12, 'Terracotta Pot 8in', 4, 2, 6.50, 2.50, 90, 25, FALSE),
  (13, 'Ceramic Planter ''Sage''', 4, 2, 19.00, 8.00, 22, 10, FALSE),
  (14, 'Hanging Basket', 4, 2, 14.00, 6.00, 0, 10, FALSE),
  (15, 'Potting Soil 20L', 5, 1, 11.00, 5.00, 75, 20, FALSE),
  (16, 'Compost Mix 10L', 5, 1, 7.50, 3.00, 48, 20, FALSE),
  (17, 'Watering Can 2gal', 6, 2, 16.00, 7.00, 33, 10, FALSE),
  (18, 'Drip Hose 25ft', 6, 5, 21.00, 10.00, 12, 8, FALSE),
  (19, 'All-Purpose Fertilizer', 7, 4, 13.50, 6.00, 40, 15, FALSE),
  (20, 'Orchid Food', 7, 4, 9.00, 4.00, 6, 10, FALSE),
  (21, 'Garden Gnome', 8, 2, 15.00, 7.00, 25, 5, FALSE),
  (22, 'Solar Lantern', 8, 2, 18.50, 9.00, 9, 8, FALSE),
  (23, 'Cactus Mix Soil', 5, 1, 9.00, 4.00, 20, 15, FALSE),
  (24, 'Bird Feeder', 8, 5, 17.00, 8.00, 4, 6, TRUE);

-- categories: Lookup table of product categories.
CREATE TABLE categories (
  category_id INTEGER,
  category_name VARCHAR(255)
);

INSERT INTO categories (category_id, category_name) VALUES
  (1, 'Houseplants'),
  (2, 'Seeds & Bulbs'),
  (3, 'Tools'),
  (4, 'Pots & Planters'),
  (5, 'Soil & Compost'),
  (6, 'Watering'),
  (7, 'Fertilizer'),
  (8, 'Outdoor Decor');

-- orders: One row per order. Unshipped orders have a null shipped_date.
CREATE TABLE orders (
  order_id INTEGER,
  customer_id INTEGER,
  order_date DATE,
  shipped_date DATE,
  status VARCHAR(255),
  coupon_code VARCHAR(255)
);

INSERT INTO orders (order_id, customer_id, order_date, shipped_date, status, coupon_code) VALUES
  (1, 1, '2024-03-02', '2024-03-04', 'shipped', 'SPRING10'),
  (2, 1, '2024-05-18', '2024-05-20', 'shipped', NULL),
  (3, 4, '2024-02-11', '2024-02-13', 'shipped', NULL),
  (4, 7, '2024-04-07', NULL, 'processing', NULL),
  (5, 3, '2024-01-25', '2024-01-28', 'shipped', 'WELCOME5'),
  (6, 6, '2024-03-15', '2024-03-18', 'shipped', NULL),
  (7, 2, '2024-05-01', NULL, 'pending', NULL),
  (8, 10, '2024-04-22', '2024-04-25', 'shipped', 'SPRING10'),
  (9, 13, '2024-05-20', NULL, 'processing', NULL),
  (10, 15, '2024-02-28', '2024-03-02', 'shipped', NULL),
  (11, 11, '2024-06-01', '2024-06-03', 'shipped', NULL),
  (12, 17, '2024-01-15', '2024-01-17', 'shipped', NULL),
  (13, 4, '2024-06-10', NULL, 'pending', NULL),
  (14, 18, '2024-03-30', '2024-04-01', 'shipped', 'WELCOME5'),
  (15, 20, '2024-05-12', '2024-05-15', 'shipped', NULL),
  (16, 7, '2024-02-19', '2024-02-22', 'shipped', NULL),
  (17, 9, '2024-04-03', NULL, 'cancelled', NULL),
  (18, 16, '2024-06-05', '2024-06-07', 'shipped', NULL),
  (19, 1, '2024-06-14', NULL, 'processing', 'SPRING10'),
  (20, 12, '2024-03-21', '2024-03-24', 'shipped', NULL),
  (21, 13, '2024-04-18', '2024-04-20', 'shipped', NULL),
  (22, 3, '2024-05-27', NULL, 'pending', NULL),
  (23, 6, '2024-06-02', '2024-06-04', 'shipped', 'SUMMER15'),
  (24, 10, '2024-01-30', '2024-02-02', 'shipped', NULL);

-- order_items: One row per line item within an order.
CREATE TABLE order_items (
  order_item_id INTEGER,
  order_id INTEGER,
  product_id INTEGER,
  quantity INTEGER,
  unit_price DECIMAL(12,2)
);

INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price) VALUES
  (1, 1, 1, 1, 32.00),
  (2, 1, 15, 2, 11.00),
  (3, 2, 2, 1, 28.00),
  (4, 2, 17, 1, 16.00),
  (5, 3, 3, 2, 18.00),
  (6, 3, 10, 1, 9.50),
  (7, 4, 6, 5, 3.50),
  (8, 4, 8, 2, 8.00),
  (9, 5, 4, 3, 12.00),
  (10, 5, 12, 4, 6.50),
  (11, 6, 5, 1, 24.00),
  (12, 6, 19, 2, 13.50),
  (13, 7, 9, 1, 22.00),
  (14, 7, 11, 2, 8.00),
  (15, 8, 2, 2, 28.00),
  (16, 8, 13, 1, 19.00),
  (17, 9, 21, 1, 15.00),
  (18, 9, 22, 1, 18.50),
  (19, 10, 1, 1, 32.00),
  (20, 10, 16, 3, 7.50),
  (21, 11, 3, 2, 18.00),
  (22, 11, 18, 1, 21.00),
  (23, 12, 7, 4, 2.75),
  (24, 12, 6, 3, 3.50),
  (25, 13, 15, 2, 11.00),
  (26, 13, 23, 1, 9.00),
  (27, 14, 5, 1, 24.00),
  (28, 14, 20, 2, 9.00),
  (29, 15, 12, 5, 6.50),
  (30, 15, 17, 1, 16.00),
  (31, 16, 2, 1, 28.00),
  (32, 16, 19, 1, 13.50),
  (33, 17, 9, 1, 22.00),
  (34, 18, 1, 2, 32.00),
  (35, 18, 3, 1, 18.00),
  (36, 19, 4, 2, 12.00),
  (37, 19, 10, 1, 9.50),
  (38, 20, 8, 3, 8.00),
  (39, 20, 16, 2, 7.50),
  (40, 21, 5, 1, 24.00),
  (41, 21, 13, 2, 19.00),
  (42, 22, 21, 1, 15.00),
  (43, 22, 22, 2, 18.50),
  (44, 23, 2, 1, 28.00),
  (45, 23, 18, 1, 21.00),
  (46, 24, 1, 1, 32.00),
  (47, 24, 15, 1, 11.00),
  (48, 24, 6, 4, 3.50);

-- suppliers: One row per supplier. Some suppliers have no contact email.
CREATE TABLE suppliers (
  supplier_id INTEGER,
  supplier_name VARCHAR(255),
  contact_email VARCHAR(255),
  state VARCHAR(255)
);

INSERT INTO suppliers (supplier_id, supplier_name, contact_email, state) VALUES
  (1, 'GreenLeaf Wholesale', 'orders@greenleaf.example', 'CA'),
  (2, 'Terra Pots Co', 'hello@terrapots.example', 'OH'),
  (3, 'RootWorks Supply', NULL, 'PA'),
  (4, 'SunGrow Nurseries', 'sales@sungrow.example', 'FL'),
  (5, 'Ironwood Tools', NULL, 'PA'),
  (6, 'Bloomfield Seeds', 'contact@bloomfield.example', 'OR');
