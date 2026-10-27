SELECT
    r.rental_id,
    r.customer_id,
    r.staff_id,
    i.store_id,
    f.film_id,
    f.length,
    f.rental_duration,
    f.rental_rate,
    f.replacement_cost,
    f.rating,
    c.name AS category,
    EXTRACT(DOW FROM r.rental_date) AS rental_weekday,
    EXTRACT(HOUR FROM r.rental_date) AS rental_hour,
    CASE
        WHEN r.return_date IS NULL THEN 1
        WHEN r.return_date > r.rental_date + (f.rental_duration * INTERVAL '1 day') THEN 1
        ELSE 0
    END AS is_late
FROM rental r
JOIN inventory i
    ON i.inventory_id = r.inventory_id
JOIN film f
    ON f.film_id = i.film_id
LEFT JOIN film_category fc
    ON fc.film_id = f.film_id
LEFT JOIN category c
    ON c.category_id = fc.category_id;