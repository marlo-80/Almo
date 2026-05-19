import pandas as pd
import os
import re
import kagglehub
import pickle

from datetime import datetime
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder,StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


kagglehub_path = "robikscube/flight-delay-dataset-20182022"

DEFAULT_PATH = "flight_data"
RANDOM_STATE = 0xdeadbeef
MODELS_DIR = "models"
MODEL_NAME = "FlightDelayPredictor.pkl"

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
                            "Year", "Month", "DayofMonth", "DayOfWeek",
                            "Airline", "Operating_Airline", "Flight_Number_Operating_Airline", "Flight_Number_Marketing_Airline",
                            "Origin", "Dest", "OriginAirportID", "DestAirportID",
                            "CRSDepTime", "CRSArrTime", "Distance", "CRSElapsedTime", #"DepDelay", #"Diverted", "Cancelled",                             
                            "ArrDelay", "Tail_Number"
                             ])        
        
        df_['CRSDepHrs']  = df_['CRSDepTime'] // 100
        df_['CRSArrHrs' ] = df_['CRSArrTime'] // 100
        df_['CRSArrMins'] = df_['CRSArrTime'] % 100
        df_['CRSDepMins'] = df_['CRSDepTime'] % 100
        df_.drop(columns=["CRSDepTime", "CRSArrTime"], inplace=True)        
        df = pd.concat([df, df_], ignore_index=True)       
    
    return df

def shuffle_dataset(big_df: pd.DataFrame, fractions : list[float], max_rows: int)-> pd.DataFrame:
    for fraction in fractions:
        # simply split randomly the dataset and keep only the fraction of it, 
        # then repeat until we have enough rows or we have exhausted the fractions list
        big_df, _  = train_test_split(big_df, test_size=fraction, random_state=RANDOM_STATE)
        if not max_rows is None and len(big_df) < max_rows and max_rows > 0:
            break
    big_df.rename(columns={ "OriginAirportID":"origin_airport_id", "DestAirportID":"dest_airport_id", 
                        "DayOfWeek":"day_of_week", "DayofMonth":"day_of_month", "CRSDepHrs":"crs_dep_hrs", "CRSDepMins":"crs_dep_mins", "CRSArrHrs":"crs_arr_hrs", 
                        "CRSElapsedTime":"crs_elapsed_time", "Flight_Number_Marketing_Airline":"flight_number_marketing_airline",   
                        "CRSArrMins":"crs_arr_mins", "ArrDelay" : "arr_delay"  }, inplace=True)
    big_df.rename (columns=lambda x : x.lower(), inplace=True)
    big_df.dropna(inplace=True)
    return big_df.reset_index(drop=True)

if __name__ == "__main__":
    print("main")
    df = load_from_local()    
    df = shuffle_dataset(df,[0.5,0.5,0.5], 250_000)
    print("shuffle done")
    
    #------------------------------
    # Model training goes here, but we will move it to a separate file later, for now we just want to test the data loading and shackle functions
    
    
    print("starting splitting")
    target_column = "arr_delay"
    df.dropna(subset=target_column, inplace=True)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print("Split done")
    
    print(f"Describe:{xTrain.describe()}")
    print(f"Info: {xTrain.info()}")
    print("splitting/replacing done")    
    print("---")

    categorical_columns = [  "origin", "dest", "origin_airport_id", "dest_airport_id", #"OriginCityName", "DestCityName",
                        "airline", "operating_airline",
                        "flight_number_marketing_airline",  
                     ]
                        
    numeric_columns = ["year", "month", "day_of_month", 'day_of_week','crs_dep_hrs', 'crs_dep_mins', 'crs_arr_hrs', 'crs_arr_mins', "distance"] #, "CRSElapsedTime",]
    print( "Building a pipeline")
    
    preprocessor = ColumnTransformer(transformers=[
        ('high_card_cat', TargetEncoder(target_type='continuous'), categorical_columns),
        #('low_card_cat', OneHotEncoder(handle_unknown='ignore'), ["Airline"], ["Operating_Airline"],
        #              ["Flight_Number_Marketing_Airline"], ["Tail_Number"]),
        ('num', StandardScaler(), numeric_columns)
    ])

    model = Pipeline([     
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=400,           # Similar to n_estimators
            max_depth=15,           # Keeps trees shallow for speed
            categorical_features=None, # It can handle categories automatically if configured
            random_state=RANDOM_STATE
        ))
    ])

    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fitting start: {time_string}")

    model.fit(xTrain, yTrain)

    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fitting done: {time_string}")


    predictions = model.predict(xTest)
    print("predictions done")

    mae = mean_absolute_error(yTest, predictions)
    r2 = r2_score(yTest, predictions)

    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    print(f"Saving model to {MODEL_NAME} using pickle...")
    # Open the file in binary write mode ('wb')
    Path(f"./{MODELS_DIR}").mkdir(parents=True, exist_ok=True)

    with open(f"./{MODELS_DIR}/{MODEL_NAME}", 'wb') as file:    
        pickle.dump(model, file)

    print(f"Model saved successfully with pickle to {MODELS_DIR}/{MODEL_NAME}")
