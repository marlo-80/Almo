{{
  config(
    materialized = 'table',
    post_hook = [
        "CHECKPOINT;",
        "ANALYZE {{ this }};"
    ]
  )
}}

WITH base_data AS (
    SELECT *,
        CONCAT(
            COALESCE("Origin", 'NA'), '_',
            COALESCE("Dest", 'NA'), '_',
            COALESCE("FlightDate"::text, '00000000'), '_',
            COALESCE("Operating_Airline", 'NA'), '_',
            COALESCE("CRSDepTime"::text, '0000')
        ) AS generated_flight_uid
    FROM {{ source('raw', 'flights') }}
    WHERE "Cancelled" IS FALSE
      AND "Diverted" IS FALSE
      AND "FlightDate" IS NOT NULL
      AND "Origin" IS NOT NULL
      AND "Dest" IS NOT NULL
      AND "ArrDelay" IS NOT NULL
      AND "ArrDelayMinutes" IS NOT NULL
      AND "ArrDel15" IS NOT NULL
      AND "Distance" > 0
),

with_uid AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY generated_flight_uid
            ORDER BY "Marketing_Airline_Network"
        ) AS rn
    FROM base_data
)

SELECT
    generated_flight_uid                AS flight_uid,
    "FlightDate"::date                  AS flight_date,
    "Year"                              AS year,
    "Quarter"                           AS quarter,
    "Month"                             AS month,
    "DayofMonth"                        AS day_of_month,
    "DayOfWeek"                         AS day_of_week,
    "Origin"                            AS origin,
    "OriginCityName"                    AS origin_city_name,
    "OriginState"                       AS origin_state,
    "OriginAirportID"                   AS origin_airport_id,
    "Dest"                              AS dest,
    "DestCityName"                      AS dest_city_name,
    "DestState"                         AS dest_state,
    "DestAirportID"                     AS dest_airport_id,
    "Distance"                          AS distance,
    "DistanceGroup"                     AS distance_group,
    "Marketing_Airline_Network"         AS marketing_airline_network,
    "Operating_Airline"                 AS operating_airline,
    "Flight_Number_Marketing_Airline"   AS flight_number_marketing_airline,
    "Flight_Number_Operating_Airline"   AS flight_number_operating_airline,
    "Tail_Number"                       AS tail_number,
    "CRSDepTime"                        AS crs_dep_time,
    "CRSArrTime"                        AS crs_arr_time,
    "CRSElapsedTime"                    AS crs_elapsed_time,
    "DepTimeBlk"                        AS dep_time_blk,
    "ArrDelay"                          AS arr_delay,
    "ArrDelayMinutes"                   AS arr_delay_minutes,
    "ArrDel15"                          AS arr_del15,
    "ArrivalDelayGroups"                AS arrival_delay_groups
FROM with_uid
WHERE rn = 1