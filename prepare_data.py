import pandas as pd
import requests
import sys
import io

from pathlib import Path

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import os
import re
import kagglehub


kagglehub_path = "robikscube/flight-delay-dataset-20182022"

DEFAULT_PATH = "flight_data"

def load_from_kaggle( kaggle_path: str, output_dir: str)->str:
    #print(f"Downloading dataset from Kaggle: {kaggle_path} to {output_dir}")
    path = kagglehub.dataset_download(kaggle_path, output_dir = output_dir, unzip = True)
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
                    usecols=["FlightDate", "Airline", "Origin", "Dest", 
                              "CRSDepTime", "CRSArrTime", "DepDelay", "Diverted", "Cancelled",
                              "Operating_Airline", "OriginCityName", "DestCityName",
                              "Tail_Number", "Flight_Number_Operating_Airline", 
                              "OriginAirportID", "DestAirportID", "DepDelayMinutes", "ArrDelay"])        
        
        df = pd.concat([df, df_], ignore_index=True)        
        
    #print(df.head())
    #print(df.describe())
    #df.info()
    return df

def shackle_dataset(big_df: pd.DataFrame, fractions : list[float], max_rows: int)-> pd.DataFrame:
    for fraction in fractions:
        # simply split randomly the dataset and keep only the fraction of it, 
        # then repeat until we have enough rows or we have exhausted the fractions list
        big_df, _  = train_test_split(big_df, test_size=fraction, random_state=0xdeadbeef)
        if not max_rows is None and len(big_df) < max_rows:
            break
    return big_df.reset_index(drop=True)


def ReplaceDateTimeWithNumeric(df: pd.DataFrame, column_name: str = "FlightDate", replace_prefix: str = "")->pd.DataFrame:
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce", format="mixed")
    df[f"{replace_prefix}Year"] = df[column_name].dt.year
    df[f"{replace_prefix}Month"] = df[column_name].dt.month
    df[f"{replace_prefix}DayOfMonth"] = df[column_name].dt.day
    df[f"{replace_prefix}DayOfWeek"] = df[column_name].dt.dayofweek
    return df.drop(columns=[column_name])

if __name__ == "__main__":
    print("main")
    df = load_from_local()

    df = shackle_dataset(df, fractions=[0.1, 0.5, 0.5], max_rows=100_000)
    df["CRSDepTime"] = pd.to_datetime(df["CRSDepTime"], errors="coerce", format="mixed")
    print(f"type of df['CRSDepTime']: {type(df['CRSDepTime'].dtype)}")
    print(f"type of df['Origin']: {type(df['Origin'].dtype)}")

    #print(df.head())
    #print(df.describe())
    #print(df.info())
    #------------------------------
    # Model training goes here, but we will move it to a separate file later, for now we just want to test the data loading and shackle functions
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    target_column = "ArrDelay"   
    
    print("starting splitting")
    target_column = "ArrDelay"
    df.dropna(subset=target_column, inplace=True)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0xdeadbeef)
    print("Split; replacing")
    xTrain = ReplaceDateTimeWithNumeric(xTrain, column_name="FlightDate")
    xTest = ReplaceDateTimeWithNumeric(xTest, column_name="FlightDate")
    xTrain['CRSDepTime'] = xTrain['CRSDepTime'].astype('int64')
    xTrain['CRSArrTime'] = xTrain['CRSArrTime'].astype('int64')

    xTest['CRSDepTime'] = xTest['CRSDepTime'].astype('int64')
    xTest['CRSArrTime'] = xTest['CRSArrTime'].astype('int64')
    
    print(xTrain.describe())
    print(xTrain.info())

    print(f"Duplicates: {xTrain.columns.duplicated().any()}")
    print(f"value_counts: {xTrain.columns.value_counts()}")
    
    #cols_list = xTrain.columns[df.isna().any()]#.tolist()
    
    #print( f"columns with nans: {cols_list}")
    print("splitting/replacing done")

    #text_columns = ["FlightDate", "Airline", "Origin", "Dest", "Tail_Number", "Operating_Airline", "OriginCityName", "DestCityName"] 
    
    numeric_columns = ["Year", "Month", "DayOfMonth", 'DayOfWeek','CRSDepTime', "CRSArrTime", "DepDelay", "Diverted", "Cancelled","DepDelayMinutes", "OriginAirportID", 
                   "DestAirportID", "Flight_Number_Operating_Airline"]
    
    preprocessor = ColumnTransformer(transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ["Airline", "Origin", "Dest"]),
    
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_columns)
])

    model = Pipeline([     
        ('preprocessor', preprocessor),
       ('regressor', LinearRegression())
    ])

    print("fitting start")
    model.fit(xTrain, yTrain)
    print("fitting done")
    
    
    predictions = model.predict(xTest)
    print("predictions done")
    mae = mean_absolute_error(yTest, predictions)
    r2 = r2_score(yTest, predictions)
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")