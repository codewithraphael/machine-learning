-- SQLBook: Code
-- film count --
SELECT
    count(*) AS total_film_count
FROM
    film;

-- first five customer details --
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email
FROM
    customer AS c
LIMIT
    5;

-- actor count --
SELECT
    count(*) AS total_actor_count
FROM
    actor;

-- distinct film rating --
SELECT DISTINCT
    rating
FROM
    film;

-- filtering PG-13 --
SELECT
    *
FROM
    film
WHERE
    rating = 'PG-13';

-- films longer than 120 minutes --
SELECT
    *
FROM
    film
WHERE
    length > 120;

-- customers whose first name starts with 'A' --
SELECT
    first_name,
    last_name
FROM
    customer
WHERE
    first_name LIKE 'A';

-- rental rate 4.99 and 'R' rating --
SELECT
    title,
    rental_rate,
    rating
FROM
    film
WHERE
    rental_rate = 4.99
    AND rating = 'R';

-- top 10 most expensive film --
SELECT
    title,
    replacement_cost
FROM
    film
ORDER BY
    replacement_cost
LIMIT
    10;

-- cheapest and most expensive film --
SELECT
    min(replacement_cost) AS cheapest,
    max(replacement_cost) AS most_expensive
FROM
    film;

-- how many films per rating --