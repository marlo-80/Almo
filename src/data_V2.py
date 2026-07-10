import os
import re
import kagglehub
import pandas as pd
import shutil
import time


from sqlalchemy import create_engine


kagglehub_path = "robikscube/flight-delay-dataset-20182022"

DEFAULT_PATH = "flight_data"

DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"



def load_from_kaggle(kaggle_path: str, output_dir: str) -> str:
    # 1. Cache in das Ausgabeverzeichnis verlegen (vermeidet volle Home‑Partition)
    cache_dir = os.path.join(output_dir, ".kaggle_cache")
    os.environ["KAGGLEHUB_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # 2. Open‑File‑Limit erhöhen (gegen "Too many open files")
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = max(8192, soft)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"Rate limit increased from {soft} to {new_soft}")
    except Exception as e:
        print(f"Rate limit can't be increased any further: {e}")

    # 3. Download mit Wiederholungen und progressivem Backoff
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Download attempts: {attempt}/{max_retries} …")
            # KEIN output_dir (kompatibel mit kagglehub 0.3.10)
            downloaded_path = kagglehub.dataset_download(kaggle_path)
            print(f"Download saved to: {downloaded_path}")

            # CSV‑Dateien in output_dir kopieren
            for file in os.listdir(downloaded_path):
                if file.endswith(".csv"):
                    src = os.path.join(downloaded_path, file)
                    dst = os.path.join(output_dir, file)
                    shutil.copy2(src, dst)
                    print(f"Copied: {file}")

            print("")
            print("Cleaning up cache...")
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("...cache deleted.")
            print("")

            return output_dir        

        except Exception as e:
            print(f"Error in attempt: {attempt}: {e}")
            # Cache aufräumen, um Platz zu sparen und beschädigte Dateien zu entfernen
            shutil.rmtree(cache_dir, ignore_errors=True)
            if attempt == max_retries:
                raise RuntimeError("Download canceled after 5 tries.")
            wait = 10 * attempt  # 10, 20, 30, 40 Sekunden
            print(f"Wait {wait} seconds before next attempts...")
            time.sleep(wait)

    raise RuntimeError("Download failed.")




def load_from_local(path = f"./{DEFAULT_PATH}/"):
    if path is None or path == "":
        path = f"./{DEFAULT_PATH}/"

    # Prüfe, ob CSV-Dateien existieren
    csv_exists = False
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".csv"):
                csv_exists = True
                break    

    if not os.path.exists(path) or not csv_exists:
        path = load_from_kaggle(kagglehub_path, output_dir=path)

    print(f"Loading data from local path: {path}")
    for file in os.listdir(path):
        try:
            print(f"checking file: {file}")
            if re.match(r"Combined_Flights_\d{4}\.csv", file) is None:
                continue
            print( f"read from file: {file}")
            df = pd.read_csv(os.path.join(path, file), 
                    usecols=[
                        # Date
                        "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
                        # Route & Distance
                        "Origin", "OriginCityName", "OriginState", "OriginAirportID", "Dest", "DestCityName", "DestState","DestAirportID", "Distance", "DistanceGroup",
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
            
            yield df        
            #df = pd.concat([df, df_], ignore_index=True)        
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}", file=sys.stderr)
            continue





def load_subset_table(query: str) -> pd.DataFrame:
    """Loads tables/views from PostgreSQL."""
    engine = create_engine(DB_URI)
    df = pd.read_sql(query, engine)
    return df




def convert_integers_to_float(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Transforms all values from int64 to float64."""
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