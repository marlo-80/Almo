--dbt/models/training/pre_covid_big.sql
{{
  config(
    materialized = 'table',
    pre_hook = "SELECT setseed(0.42);",
    post_hook = [
        "CHECKPOINT;",
        "ANALYZE {{ this }};"
    ],
    indexes = [
      {'columns': ['flight_date'], 'type': 'btree'}
    ]
  )
}}



WITH randomized AS (
    SELECT *
    FROM {{ ref('stg_flights') }}
    WHERE flight_date >= '2018-01-01'
      AND flight_date <  '2019-12-31'
      AND random() < 0.04                  -- Umcomment this to improve performance
    ORDER BY random()
)
SELECT *
FROM randomized
ORDER BY flight_date

