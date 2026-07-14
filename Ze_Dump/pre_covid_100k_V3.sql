{{
  config(
    materialized = 'table',
    indexes = [
      {'columns': ['flight_date'], 'type': 'btree'}
    ],
    post_hook = [
        "CHECKPOINT;",
        "ANALYZE {{ this }};"
    ]
  )
}}

SELECT *
FROM {{ ref('stg_flights') }}
WHERE flight_date >= '2018-01-01'
  AND flight_date <  '2019-12-31'
  AND ABS(hashtext(flight_uid) % 1000) < 10  -- ~0.4% sample
ORDER BY flight_date
LIMIT 100000