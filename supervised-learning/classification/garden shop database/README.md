# Garden Shop Database and ML Project

A small relational sales database for practicing SQL, data preparation, exploratory analysis, and the early stages of a machine-learning workflow. The source dataset is in [`database/garden_shop.sql`](database/garden_shop.sql).

## Project Goals

- Explore customer, order, product, and inventory data with SQL.
- Build order- and customer-level features from normalized tables.
- Use the project as a starting point for a carefully evaluated retail ML experiment.

This repository currently contains the SQL dataset only; model training and evaluation are future work.

## Database

The SQL script creates and populates six tables (130 rows in total):

| Table | Rows | Contents |
| --- | ---: | --- |
| `customers` | 20 | Customer contact details, location, signup date, and active flag |
| `orders` | 24 | Order date, shipping date, status, coupon, and customer reference |
| `order_items` | 48 | Products and quantities purchased in each order, at the order-time unit price |
| `products` | 24 | Product category and supplier references, price, cost, and stock levels |
| `categories` | 8 | Product category names |
| `suppliers` | 6 | Supplier names, contact email, and state |

The intended relationships are `customers` to `orders`, `orders` to `order_items`, `products` to `order_items`, `categories` to `products`, and `suppliers` to `products`. These relationships are represented by ID columns, but the script does not declare primary-key or foreign-key constraints.

## Load the Database

Use PostgreSQL's command-line tools. Make sure the PostgreSQL server is running and `createdb` and `psql` are available in your terminal. From this project directory, create the database and load the schema and sample data:

```powershell
createdb garden_shop
psql -d garden_shop -f "database/garden_shop.sql"
```

If your PostgreSQL role or connection settings differ from your Windows username and local defaults, pass them to both commands, for example `-U your_username -h localhost -p 5432`. The script drops and recreates its six tables each time it runs. Do not run it against a database containing data you need to keep. The SQL is also compatible with MySQL, SQLite, and DuckDB; SQL Server requires the type and boolean changes noted at the top of the script.

## Example Analysis

This query reports revenue and units sold by product category. Revenue here is the sum of `order_items.quantity * order_items.unit_price`; it does not apply coupon discounts, refunds, or taxes because those details are not present in the dataset.

```sql
SELECT
    c.category_name,
    SUM(oi.quantity) AS units_sold,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) AS gross_line_revenue
FROM order_items AS oi
JOIN products AS p ON p.product_id = oi.product_id
JOIN categories AS c ON c.category_id = p.category_id
JOIN orders AS o ON o.order_id = oi.order_id
WHERE o.status <> 'cancelled'
GROUP BY c.category_id, c.category_name
ORDER BY gross_line_revenue DESC;
```

## ML Direction

A reasonable supervised-learning extension is to predict whether a customer places another order within a defined future window, such as 60 days. Create customer features using only data available at a chosen prediction date (for example, prior order count, days since last order, prior spend, and product-category mix), then label each customer based on orders in the following 60 days. Exclude identifiers and any information from after the prediction date to prevent leakage.

The included data is too small to support a reliable predictive model: it has only 20 customers and 24 orders, covering roughly January through June 2024. The customer `is_active` field is present, but its meaning and labeling date are unspecified, so it should not automatically be treated as a trustworthy prediction target. Use this dataset to practice joins, feature construction, and validation design; collect substantially more time-stamped transactions and define the target clearly before drawing model conclusions. A larger history would also be needed for useful demand forecasting or product recommendations.

## Suggested Workflow

1. Load the SQL script and verify the table counts.
2. Inspect nulls, status values, date ranges, and relationships; decide how cancelled and unshipped orders should be handled.
3. Write SQL queries for sales, repeat purchasing, shipping time, and inventory availability.
4. Export or query a feature table at the customer or product level, keeping a clear prediction cutoff.
5. Once more labeled data is available, establish a simple baseline and use a time-based validation split before comparing models.

## Dataset Notes

- Missing phone numbers and some supplier emails are intentional in the sample.
- `orders.shipped_date` is null for orders that have not shipped.
- `products.quantity_on_hand` is a snapshot; the database does not include historical inventory levels.
- `order_items.unit_price` records the line-item price, while product `price` is the listed product price.
- The SQL file includes its source and reuse note.