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
SELECT
    c.name,
    f.title
FROM
    category c
    JOIN film_category fc ON c.category_id = fc.category_id
    JOIN film f ON fc.film_id = f.film_id;

-- show ratings where average rental is higher than 3.00 --
SELECT
    rating,
    round(avg(rental_rate), 2) AS avg_rental_rate
FROM
    film
GROUP BY
    rating
HAVING
    avg(rental_rate) > 3.00;

-- matching rows from film and language tables --
SELECT
    f.title,
    l.name
FROM
    film f
    INNER JOIN LANGUAGE l ON f.language_id = l.language_id
LIMIT
    10;

-- film title corresponding to each actor name --
SELECT
    a.first_name,
    a.last_name,
    f.title
FROM
    actor a
    JOIN film_actor fa ON a.actor_id = fa.actor_id
    JOIN film f ON f.film_id = fa.film_id;

-- total amount paid by customers --
SELECT
    concat (c.first_name, ' ', c.last_name) AS full_name,
    sum(p.amount) AS total_amount_paid
FROM
    customer c
    JOIN payment p ON c.customer_id = p.customer_id
GROUP BY
    c.customer_id,
    c.first_name,
    c.last_name
ORDER BY
    total_amount_paid DESC
LIMIT
    10;

-- store address with staff name --
SELECT
    concat (s.first_name, ' ', s.last_name) AS staff_name,
    a.address,
    a.district,
    a.city_id,
    a.postal_code
FROM
    staff s
    JOIN address a ON s.address_id = a.address_id
ORDER BY
    staff_name;

-- total number of films in each category --
SELECT
    c.name AS category_name,
    count(f.film_id) AS total_films
FROM
    category c
    JOIN film_category fc ON c.category_id = fc.category_id
    LEFT JOIN film f ON f.film_id = fc.film_id
GROUP BY
    category_name;

-- top l0 rented films --
SELECT
    title,
    max(rental_rate)
FROM
    film
GROUP BY
    title
LIMIT
    10;
    
--- rental events by month ---

select DATE_TRUNC('month', rental_date)::date as rental_month,
count(*) as rentals
from rental
GROUP BY 1
ORDER BY 1;