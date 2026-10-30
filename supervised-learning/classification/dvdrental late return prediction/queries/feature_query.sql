-- Customer-level classification dataset.
-- The label is 1 when the customer rents another movie within 30 days
-- of the current rental. Change the interval to use another prediction horizon.
--
-- Every feature is calculated from rentals before the current rental, so the
-- query can be used to train a model without using future rental activity.

WITH rental_history AS (
    SELECT
        r.rental_id,
        r.customer_id,
        r.inventory_id,
        r.rental_date,
        MAX(r.rental_date) OVER () AS dataset_end_date,
        LAG(r.rental_date) OVER (
            PARTITION BY r.customer_id
            ORDER BY r.rental_date, r.rental_id
        ) AS previous_rental_date,
        LEAD(r.rental_date) OVER (
            PARTITION BY r.customer_id
            ORDER BY r.rental_date, r.rental_id
        ) AS next_rental_date,
        COUNT(*) OVER (
            PARTITION BY r.customer_id
            ORDER BY r.rental_date, r.rental_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_rental_count
    FROM rental AS r
),
rental_gaps AS (
    SELECT
        rh.*,
        EXTRACT(
            EPOCH FROM (rh.rental_date - rh.previous_rental_date)
        ) / 86400.0 AS days_since_previous_rental
    FROM rental_history AS rh
),
customer_features AS (
    SELECT
        rg.*,
        AVG(rg.days_since_previous_rental) OVER (
            PARTITION BY rg.customer_id
            ORDER BY rg.rental_date, rg.rental_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_average_days_between_rentals
    FROM rental_gaps AS rg
)
SELECT
    cf.rental_id,
    cf.customer_id,
    c.store_id,
    i.film_id,
    f.rating,
    f.rental_duration,
    f.rental_rate,
    f.length AS film_length,
    cf.rental_date,
    EXTRACT(DOW FROM cf.rental_date)::integer AS rental_weekday,
    EXTRACT(HOUR FROM cf.rental_date)::integer AS rental_hour,
    cf.prior_rental_count,
    ROUND(cf.days_since_previous_rental::numeric, 2)
        AS days_since_previous_rental,
    ROUND(cf.prior_average_days_between_rentals::numeric, 2)
        AS prior_average_days_between_rentals,
    CASE
        WHEN cf.next_rental_date <= cf.rental_date + INTERVAL '30 days'
            THEN 1
        ELSE 0
    END AS rented_again_within_30_days
FROM customer_features AS cf
JOIN customer AS c
    ON c.customer_id = cf.customer_id
JOIN inventory AS i
    ON i.inventory_id = cf.inventory_id
JOIN film AS f
    ON f.film_id = i.film_id
WHERE cf.rental_date <= cf.dataset_end_date - INTERVAL '30 days'
ORDER BY cf.customer_id, cf.rental_date, cf.rental_id;