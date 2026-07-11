# ./docker/src/data.py
"""
Data Loading Module – Kaggle Download & PostgreSQL Integration

This module handles all data loading operations for the flight delay
prediction pipeline. It provides functions to:

- Download the raw flight delay dataset from Kaggle (via `kagglehub`)
- Cache and clean up downloaded files to save disk space
- Load CSV files from local storage into Pandas DataFrames
- Read tables/views from PostgreSQL
- Convert integer columns to float for ML compatibility

The module is designed to be idempotent and resilient: downloads are
retried with exponential backoff, and existing data is reused when possible.

------------------------------------------------------------------------------
File Location
------------------------------------------------------------------------------
- Host:   ./docker/src/data.py
- Container: /app/src/data.py

------------------------------------------------------------------------------
Dependencies
------------------------------------------------------------------------------
- kagglehub     : Dataset download from Kaggle
- pandas        : Data loading and manipulation
- sqlalchemy    : PostgreSQL connection
- resource      : File descriptor limit management (Unix only)

------------------------------------------------------------------------------
Key Functions
------------------------------------------------------------------------------
- load_from_kaggle()     : Downloads and caches the Kaggle dataset
- load_from_local()      : Loads CSV files from local `flight_data/` directory
- load_subset_table()    : Reads any PostgreSQL table/view into a DataFrame
- convert_integers_to_float() : Converts int64 columns to float64
"""

import os
import re
import sys
import kagglehub
import pandas as pd
import shutil
import time
from sqlalchemy import create_engine


kagglehub_path = "robikscube/flight-delay-dataset-20182022"

DEFAULT_PATH = "flight_data"

DB_URI = "postgresql://testuser:testuser@postgres:5432/fastapi_db"



def load_from_kaggle(kaggle_path: str, output_dir: str) -> str:
    """
    Download the Kaggle dataset and copy CSV files to the output directory.

    The function sets a custom cache directory to avoid filling the home
    partition, increases the open-file limit to prevent errors during large
    downloads, and retries up to 5 times with exponential backoff.

    After downloading, both the custom cache and the temporary download folder
    are deleted to free disk space.

    Args:
        kaggle_path (str): Kaggle dataset identifier
                           (e.g., "robikscube/flight-delay-dataset-20182022").
        output_dir (str): Local directory where CSV files will be copied.

    Returns:
        str: The output directory path (same as `output_dir`).

    Raises:
        RuntimeError: If download fails after 5 retry attempts.
    """

    # 1. Move kaggle cache to project folder. Prevents full disc problems on system partitions
    cache_dir = os.path.join(output_dir, ".kaggle_cache")
    os.environ["KAGGLEHUB_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # 2. Increase open file limit (prevents "Too many open files")
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = max(8192, soft)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"Rate limit increased from {soft} to {new_soft}")
    except Exception as e:
        print(f"Rate limit can't be increased any further: {e}")

    # 3. Download with retries and progressive backoff
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Download attempts: {attempt}/{max_retries} …")
            # No output_dir (compatible with kagglehub 0.3.10)
            downloaded_path = kagglehub.dataset_download(kaggle_path)
            print(f"Download saved to: {downloaded_path}")

            # Copy CSV files to output_dir 
            for file in os.listdir(downloaded_path):
                if file.endswith(".csv"):
                    src = os.path.join(downloaded_path, file)
                    dst = os.path.join(output_dir, file)
                    shutil.copy2(src, dst)
                    print(f"Copied: {file}")

            print("")
            print("Cleaning up download folder...")
            shutil.rmtree(cache_dir, ignore_errors=True)
            shutil.rmtree(downloaded_path, ignore_errors=True)
            print("...downloads deleted.")
            print("")

            return output_dir        

        except Exception as e:
            print(f"Error in attempt: {attempt}: {e}")
            # Cleanup cache
            shutil.rmtree(cache_dir, ignore_errors=True)
            if attempt == max_retries:
                raise RuntimeError("Download canceled after 5 tries.")
            wait = 10 * attempt  # 10, 20, 30, 40 Sekunden
            print(f"Wait {wait} seconds before next attempts...")
            time.sleep(wait)

    raise RuntimeError("Download failed.")




def load_from_local(path = f"./{DEFAULT_PATH}/"):
    """
    Load CSV files from a local directory into a generator of DataFrames.

    The function checks if CSV files exist in the given path. If not, it
    triggers `load_from_kaggle()` to download the dataset. Each matched CSV
    file is read using the fast PyArrow engine and yields a DataFrame.

    The function matches files with names matching the pattern:
        Combined_Flights_<4-digit-year>.csv

    Args:
        path (str): Directory path containing CSV files.
                    Default: "./flight_data/".

    Yields:
        pd.DataFrame: One DataFrame per matching CSV file.

    Raises:
        Various exceptions from `pd.read_csv` are caught and printed to stderr,
        but do not stop the generator.
    """
    if path is None or path == "":
        path = f"./{DEFAULT_PATH}/"

    # Check for CSV files
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
                    ],
                    engine="pyarrow",
                    dtype_backend="pyarrow"                   
                    )       
            
            yield df        
            #df = pd.concat([df, df_], ignore_index=True)        
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}", file=sys.stderr)
            continue





def load_subset_table(query: str) -> pd.DataFrame:
    """
    Load a table or view from PostgreSQL into a pandas DataFrame.

    This is a convenience wrapper around `pd.read_sql` using the default
    database connection URI.

    Args:
        query (str): SQL query to execute (e.g., "SELECT * FROM dbt_staging.retrain").

    Returns:
        pd.DataFrame: Result set as a DataFrame.
    """
    engine = create_engine(DB_URI)
    df = pd.read_sql(query, engine)
    return df




def convert_integers_to_float(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Convert integer columns to float64 for compatibility with ML models.

    Many scikit-learn transformers expect float inputs. This function safely
    converts only the columns specified in `numeric_cols` that are of dtype
    `int64`.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_cols (list[str]): List of column names to convert.

    Returns:
        pd.DataFrame: A copy of the DataFrame with converted columns.
    """
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'int64':
            df[col] = df[col].astype('float64')
    return df




# Wird vermutlich nicht mehr gebraucht. Checken und entfernen:
def shackle_dataset(big_df: pd.DataFrame, fractions : list[float], max_rows: int)-> pd.DataFrame:
    """
    [DEPRECATED] Randomly reduce dataset size in steps.

    This function is no longer used in the active pipeline. It was originally
    intended to reduce dataset size by applying a series of random splits.

    Args:
        big_df (pd.DataFrame): Input DataFrame.
        fractions (list[float]): List of fractions to split off at each step.
        max_rows (int): Stop when the dataset reaches this row count.

    Returns:
        pd.DataFrame: Reduced DataFrame with reset index.

    Note:
        This function is deprecated and should be removed in a future cleanup.
        It references an undefined `train_test_split` (from sklearn) and is not
        imported anywhere in the current codebase.
    """
    for fraction in fractions:
        # simply split randomly the dataset and keep only the fraction of it, 
        # then repeat until we have enough rows or we have exhausted the fractions list
        big_df, _  = train_test_split(big_df, test_size=fraction, random_state=0xdeadbeef)
        if not max_rows is None and len(big_df) < max_rows:
            break
    return big_df.reset_index(drop=True)