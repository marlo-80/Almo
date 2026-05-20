{{
  config(
    materialized = 'table',
    indexes = [
      {'columns': ['flight_date'], 'type': 'btree'}
    ]
  )
}}

WITH randomized AS (
    SELECT *
    FROM {{ ref('stg_flights') }}
    WHERE flight_date >= '{{ var("start_date") }}'
      AND flight_date <  '{{ var("end_date") }}'
    ORDER BY random()
    LIMIT {{ var("sample_size") }}
)
SELECT *
FROM randomized
ORDER BY flight_date