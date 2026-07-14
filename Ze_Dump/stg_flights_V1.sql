{{
  config(
    materialized = 'table',
    post_hook = "CHECKPOINT;"
  )
}}

WITH with_uid AS (
    SELECT *,
        CONCAT(
            COALESCE("Origin", 'NA'), '_',
            COALESCE("Dest", 'NA'), '_',
            TO_CHAR("FlightDate"::date, 'YYYYMMDD'), '_',
            COALESCE("Operating_Airline", 'NA'), '_',
            COALESCE("CRSDepTime"::text, '0000')
        ) AS flight_uid,
        ROW_NUMBER() OVER (
            PARTITION BY CONCAT(
                COALESCE("Origin", 'NA'), '_',
                COALESCE("Dest", 'NA'), '_',
                TO_CHAR("FlightDate"::date, 'YYYYMMDD'), '_',
                COALESCE("Operating_Airline", 'NA'), '_',
                COALESCE("CRSDepTime"::text, '0000')
            )
            ORDER BY "Marketing_Airline_Network"
        ) AS rn
    FROM {{ source('raw', 'flights') }}
    WHERE "Cancelled" IS FALSE
      AND "Diverted" IS FALSE
      AND "FlightDate" IS NOT NULL
      AND "Year" IS NOT NULL                      
      AND "Quarter" IS NOT NULL                        
      AND "Month" IS NOT NULL                           
      AND "DayofMonth" IS NOT NULL                        
      AND "DayOfWeek" IS NOT NULL                        
      AND "CRSDepTime" IS NOT NULL
      AND "CRSArrTime" IS NOT NULL
      AND "CRSElapsedTime" IS NOT NULL
      AND "Origin" IS NOT NULL
      AND "Dest" IS NOT NULL
      AND "Distance" > 0
      AND "ArrDelay" IS NOT NULL
      AND "ArrDelayMinutes" IS NOT NULL
      AND "ArrDel15" IS NOT NULL
      AND "ArrivalDelayGroups" IS NOT NULL
)
SELECT
    flight_uid,
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