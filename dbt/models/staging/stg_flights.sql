{{
  config(
    materialized = 'view'
  )
}}

WITH cleaned AS (
    SELECT DISTINCT *          -- removes doubled rows
    FROM {{ source('raw', 'flights') }}
    WHERE "Cancelled" IS FALSE
      AND "Diverted" IS FALSE
      AND "CRSDepTime" IS NOT NULL
      AND "Origin" IS NOT NULL
      AND "Dest" IS NOT NULL
      AND "Distance" > 0       
)
SELECT
    -- Date
    "FlightDate"::date                  AS flight_date,
    "Year"                              AS year,
    "Quarter"                           AS quarter,
    "Month"                             AS month,
    "DayofMonth"                        AS day_of_month,
    "DayOfWeek"                         AS day_of_week,
    -- Route & Distance
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
    -- Airline & Flight
    "Marketing_Airline_Network"         AS marketing_airline_network,
    "Operating_Airline"                 AS operating_airline,
    "Flight_Number_Marketing_Airline"   AS flight_number_marketing_airline,
    "Flight_Number_Operating_Airline"   AS flight_number_operating_airline,
    "Tail_Number"                       AS tail_number,
    -- Planed Departure and Arrivals
    "CRSDepTime"                        AS crs_dep_time,
    "CRSArrTime"                        AS crs_arr_time,
    "CRSElapsedTime"                    AS crs_elapsed_time,
    -- Daytime Bins
    "DepTimeBlk"                        AS dep_time_blk,
    -- Targets
    "ArrDelay"                          AS arr_delay,
    "ArrDelayMinutes"                   AS arr_delay_minutes,
    "ArrDel15"                          AS arr_del15,
    "ArrivalDelayGroups"                AS arrival_delay_groups,
    "DepDelay"                          AS dep_delay,
    "DepDelayMinutes"                   AS dep_delay_minutes
FROM cleaned