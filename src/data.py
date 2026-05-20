import os
import re
import kagglehub
import pandas as pd

from sqlalchemy import create_engine


kagglehub_path = "robikscube/flight-delay-dataset-20182022"

DEFAULT_PATH = "flight_data"

DB_URI = "postgresql://vikmar:vikmar@postgres:5432/fastapi_db"



def load_from_kaggle( kaggle_path: str, output_dir: str)->str:
    #print(f"Downloading dataset from Kaggle: {kaggle_path} to {output_dir}")
    path = kagglehub.dataset_download(kaggle_path, output_dir = output_dir)
    print("Path to dataset files:", path)
    return path




def load_from_local(path = f"./{DEFAULT_PATH}/")->pd.DataFrame:
    if path is None or path == "":
        path = f"./{DEFAULT_PATH}/"

    if not os.path.exists(path) or os.listdir(path) == []:
        path = load_from_kaggle(kagglehub_path, output_dir=path)

    df = pd.DataFrame()
    for file in os.listdir(path):
        print(f"checking file: {file}")
        if re.match(r"Combined_Flights_\d{4}\.csv", file) is None:
            continue
        print( f"read from file: {file}")
        df_ = pd.read_csv(os.path.join(path, file), 
                    usecols=[
                        # Date
                        "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
                        # Route & Distance
                        "Origin", "OriginCityName", "OriginState", "Dest", "DestCityName", "DestState", "Distance", "DistanceGroup",
                        # Airline & Flight
                        "Marketing_Airline_Network", "Operating_Airline", "Flight_Number_Marketing_Airline", "Flight_Number_Operating_Airline", "Tail_Number",
                        # Planed Departure and Arrivals
                        "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
                        # Daytime Bins
                        "DepTimeBlk",
                        # Targets
                        "ArrDelay", "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups", "DepDelay", "DepDelayMinutes",
                        # Filtering (needed for WHERE clause)
                        "Cancelled", "Diverted"
                    ])        
        
        df = pd.concat([df, df_], ignore_index=True)        
        
    #print(df.head())
    #print(df.describe())
    #df.info()
    return df





def load_subset_table(query: str) -> pd.DataFrame:
    """Lädt eine beliebige Tabelle/View aus PostgreSQL."""
    engine = create_engine(DB_URI)
    df = pd.read_sql(query, engine)
    return df




def convert_integers_to_float(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Konvertiert alle als int64 gespeicherten numerischen Spalten nach float64."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'int64':
            df[col] = df[col].astype('float64')
    return df




# Wird vermutlich nicht mehr gebraucht. Checken und entfernen:
def shackle_dataset(big_df: pd.DataFrame, fractions : list[float], max_rows: int)-> pd.DataFrame:
    for fraction in fractions:
        # simply split randomly the dataset and keep only the fraction of it, 
        # then repeat until we have enough rows or we have exhausted the fractions list
        big_df, _  = train_test_split(big_df, test_size=fraction, random_state=0xdeadbeef)
        if not max_rows is None and len(big_df) < max_rows:
            break
    return big_df.reset_index(drop=True)