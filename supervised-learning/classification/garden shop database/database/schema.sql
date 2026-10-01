-- Run garden_shop.sql first to create and populate these tables.

ALTER TABLE customers
    ADD CONSTRAINT pk_customers PRIMARY KEY (customer_id);

ALTER TABLE categories
    ADD CONSTRAINT pk_categories PRIMARY KEY (category_id);

ALTER TABLE suppliers
    ADD CONSTRAINT pk_suppliers PRIMARY KEY (supplier_id);

ALTER TABLE products
    ADD CONSTRAINT pk_products PRIMARY KEY (product_id);

ALTER TABLE orders
    ADD CONSTRAINT pk_orders PRIMARY KEY (order_id);

ALTER TABLE order_items
    ADD CONSTRAINT pk_order_items PRIMARY KEY (order_item_id);

ALTER TABLE products
    ADD CONSTRAINT fk_products_categories
    FOREIGN KEY (category_id) REFERENCES categories (category_id);

ALTER TABLE products
    ADD CONSTRAINT fk_products_suppliers
    FOREIGN KEY (supplier_id) REFERENCES suppliers (supplier_id);

ALTER TABLE orders
    ADD CONSTRAINT fk_orders_customers
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id);

ALTER TABLE order_items
    ADD CONSTRAINT fk_order_items_orders
    FOREIGN KEY (order_id) REFERENCES orders (order_id);

ALTER TABLE order_items
    ADD CONSTRAINT fk_order_items_products
    FOREIGN KEY (product_id) REFERENCES products (product_id);