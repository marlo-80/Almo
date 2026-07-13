{{
  config(
    materialized = 'table',
    pre_hook = [
      "SET work_mem = '512MB';",
      "DO $$ DECLARE row_count bigint; sample_percent numeric; BEGIN SELECT COUNT(*) INTO row_count FROM {{ ref('stg_flights') }}; sample_percent := (100000::numeric / row_count) * 100; PERFORM set_config('my.sample_percent', sample_percent::text, false); END $$;"
    ],
    indexes = [
      {'columns': ['flight_date'], 'type': 'btree'}
    ]
  )
}}

WITH sampled AS (
    SELECT *
    FROM {{ ref('stg_flights') }}
    TABLESAMPLE SYSTEM(current_setting('my.sample_percent')::numeric)
    WHERE flight_date >= '2020-01-01'
      AND flight_date <  '2022-12-31'
)
SELECT *
FROM sampled
ORDER BY flight_date
LIMIT 100000