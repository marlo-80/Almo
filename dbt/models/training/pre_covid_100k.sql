{{
  config(
    materialized = 'table',
    indexes = [
      {'columns': ['flight_date'], 'type': 'btree'},
      {'columns': ['flight_uid'], 'type': 'btree'}
    ],
    post_hook = [
        "ANALYZE {{ this }};"
    ]
  )
}}

WITH pre_sorted_sample AS (
    -- Nur die minimale Spaltenmenge für die schwere Arbeit
    SELECT 
        flight_uid,
        flight_date
    FROM {{ ref('stg_flights') }}
    WHERE flight_date >= '2018-01-01'
      AND flight_date <  '2019-12-31'
      AND ABS(hashtext(flight_uid) % 1000) < 10  -- ~0.4% Sample
    ORDER BY flight_date
    LIMIT 100000
)

-- Alle 61 Spalten werden erst nach der Reduktion per JOIN geholt
SELECT 
    base.*
FROM {{ ref('stg_flights') }} base
JOIN pre_sorted_sample sample 
  ON base.flight_uid = sample.flight_uid
ORDER BY base.flight_date