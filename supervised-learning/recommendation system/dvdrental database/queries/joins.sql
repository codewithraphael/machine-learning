select
    a.first_name as actor_first_name,
    a.last_name as actor_last_name,
    f.title as movie_title,
    f.description,
    f.release_year,
    f.rental_duration,
    f.rental_rate,
    f.length,
    f.replacement_cost,
    f.rating,
    l.name as language
from 
    actor a
join film_actor as fa
    on a.actor_id = fa.actor_id
join film as f
    on fa.film_id = f.film_id
join language as l
    on l.language_id = f.language_id;

select
    c.customer_id,
    c.first_name,
    c.last_name,
    c.create_date,
    c.last_update,
    f.title as movie_title,
    r.rental_date,
    r.return_date,
    r.last_update,
    c.active
from customer c
join rental r
    on c.customer_id = r.customer_id
join payment p
    on p.rental_id = r.rental_id
join inventory i
    on i.inventory_id = r.inventory_id
join film f
    on f.film_id = i.film_id