WITH source AS (
    SELECT * FROM raw.flights
)
SELECT
    "FlightDate"::date                  AS flight_date,
    "Airline"                           AS airline,
    "Origin"                            AS origin,
    "Dest"                              AS dest,
    "CRSDepTime"                        AS crs_dep_time,
    "CRSArrTime"                        AS crs_arr_time,
    "DepDelay"                          AS dep_delay,
    "ArrDelay"                          AS arr_delay,
    "DepDelayMinutes"                   AS dep_delay_minutes,
    "Cancelled"                         AS is_cancelled,
    "Diverted"                          AS is_diverted,
    "Operating_Airline"                 AS operating_airline,
    "OriginCityName"                    AS origin_city,
    "DestCityName"                      AS dest_city,
    "Tail_Number"                       AS tail_number,
    "Flight_Number_Operating_Airline"   AS flight_number,
    "OriginAirportID"                   AS origin_airport_id,
    "DestAirportID"                     AS dest_airport_id
FROM source
WHERE "Cancelled" IS FALSE
  AND "Diverted" IS FALSE