import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_QUERY = '''

SELECT
    r.rental_id,
    r.customer_id,
    r.rental_date,
    r.return_date,
    f.film_id,
    f.length,
    f.rental_duration,
    f.rental_rate,
    f.replacement_cost,
    f.rating,
    c.name AS category,

    EXTRACT(
        DAY
        FROM
            r.rental_date
    ) AS weekday,
    EXTRACT(
        HOUR
        FROM
            r.rental_date
    ) AS hour,

    CASE
        WHEN r.return_date IS NULL THEN 1 -- not returned = late
        WHEN r.return_date > rental_date + f.rental_duration * INTERVAL '1 day' THEN 1
        ELSE 0
    END AS is_late
FROM rental r
join inventory i ON r.inventory_id = i.inventory_id
join film f on i.film_id = f.film_id
join film_category fc on f.film_id = fc.film_id
join category c on fc.category_id = c.category_id


'''

