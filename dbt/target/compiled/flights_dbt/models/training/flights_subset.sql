

WITH randomized AS (
    SELECT *
    FROM "fastapi_db"."dbt_staging"."stg_flights"
    WHERE flight_date >= '2019-01-01'
      AND flight_date <  '2020-01-01'
    ORDER BY random()
    LIMIT 100000
)
SELECT *
FROM randomized
ORDER BY flight_date