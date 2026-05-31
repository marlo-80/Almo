--dbt/models/training/pre_covid_big.sql
{{
  config(
    materialized = 'table',
    pre_hook = "SELECT setseed(0.42);",
    indexes = [
      {'columns': ['flight_date'], 'type': 'btree'}
    ]
  )
}}



WITH randomized AS (
    SELECT *
    FROM {{ ref('stg_flights') }}
    WHERE flight_date >= '2018-01-01'
      AND flight_date <  '2020-01-01'
      AND random() < 0.04                  -- Umcomment this to improve performance
    ORDER BY random()
    LIMIT 1000000
)
SELECT *
FROM randomized
ORDER BY flight_date

