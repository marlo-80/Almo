{{
  config(
    materialized = 'table',
    indexes = [
      {'columns': ['flight_date'], 'type': 'btree'}
    ]
  )
}}

SELECT *
FROM {{ ref('stg_flights') }}
WHERE flight_date >= '2020-01-01'
  AND flight_date <  '2022-12-31'
  AND ABS(hashtext(flight_uid) % 1000) < 10  -- ~0.4% sample
ORDER BY flight_date
LIMIT 100000