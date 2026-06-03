import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://vikmar:vikmar@postgres:5432/fastapi_db")

# Spalteninformationen aus der Staging-View abfragen
with engine.connect() as conn:
    cols = conn.execute(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'dbt_staging' AND table_name = 'stg_flights'
        ORDER BY ordinal_position
    """)).fetchall()

# Mapping von PostgreSQL-Datentypen zu SQL-Casts für JSONB-Extraktion
type_cast_map = {
    'integer': 'int',
    'bigint': 'int',
    'smallint': 'int',
    'double precision': 'float',
    'real': 'float',
    'numeric': 'float',
    'text': 'text',
    'character varying': 'text',
    'date': 'date',
    'timestamp without time zone': 'timestamp',
    'timestamp with time zone': 'timestamptz',
    'boolean': 'boolean',
}

# Sonderbehandlung für bekannte Spalten
special_cols = {
    'flight_uid': "flight_uid",
    'flight_date': "timestamp::date",
    'arr_delay_minutes': "(ground_truth->>'arr_delay_minutes')::float",
    'arr_del15': "(ground_truth->>'arr_del15')::int",
}

# Baue die SELECT-Ausdrücke für den Predictions-Teil
select_exprs = []
for col_name, col_type in cols:
    if col_name in special_cols:
        expr = special_cols[col_name]
    else:
        pg_type = col_type.lower()
        cast = type_cast_map.get(pg_type, 'text')
        expr = f"(input_features->>'{col_name}')::{cast}"
    select_exprs.append(f"{expr} AS {col_name}")

predictions_sql = "SELECT " + ",\n       ".join(select_exprs) + """
FROM api.predictions
WHERE ground_truth IS NOT NULL"""

# Gesamte View zusammenbauen
view_sql = f"""
CREATE OR REPLACE VIEW dbt_staging.training_with_drift AS
SELECT * FROM dbt_staging.stg_flights
UNION ALL
{predictions_sql}
"""

with engine.connect() as conn:
    conn.execute(text(view_sql))
    conn.commit()

print("View dbt_staging.training_with_drift wurde erstellt.")