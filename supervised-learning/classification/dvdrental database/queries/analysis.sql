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
SELECT
    rating,
    count(*) AS film_count
FROM
    film
GROUP BY
    rating
ORDER BY
    film_count DESC;

-- average length of film by rating --
SELECT
    rating,
    ROUND(AVG(length)) AS average_length
FROM
    film
GROUP BY
    rating
ORDER BY
    average_length DESC;

-- rating with more than 200 films --
SELECT
    rating,
    count(*) AS number_of_films
FROM
    film
GROUP BY
    rating
HAVING
    count(*) > 200
ORDER BY
    number_of_films DESC;

-- customers count in each store --
SELECT
    store_id,
    count(*) AS customer_count
FROM
    customer
GROUP BY
    store_id
ORDER BY
    customer_count DESC;

-- total payment receieved by each staff --
SELECT
    s.staff_id,
    sum(p.amount) AS total_payment
FROM
    staff s
    JOIN payment p ON s.staff_id = p.staff_id
GROUP BY
    s.staff_id
ORDER BY
    total_payment;

-- average rental duration in days for each rating (use rental, inventory + film or just simple with film's rental duration) --
SELECT
    rating,
    round(avg(rental_duration), 2) AS avg_rental_duration
FROM
    film
GROUP BY
    rating
ORDER BY
    avg_rental_duration DESC;

-- categories with more than 60 films --

select
    c.name,
    f.title
from category c
join film_category fc
    on c.category_id = fc.category_id
join film f
    on fc.film_id = f.film_id


