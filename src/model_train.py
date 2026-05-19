import pandas as pd

from prepare_data import load_from_local, shuffle_dataset, MODELS_DIR, MODEL_NAME, RANDOM_STATE
import pickle

from datetime import datetime
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def LoadData():
    df = load_from_local()
    print(f"Dataset loaded with shape: {df.shape}")
    df = shuffle_dataset(df,[0.5,0.5,0.5], max_rows=2_500_000)
    print("Dataset shuffled")
    
    return df

def CreateModel(df: pd.DataFrame, *, categorical_columns: list[str] = [], numeric_columns: list[str] = [], target_column: str ):
    print(f"Dataset loaded with shape: {df.shape}")    

    X: pd.DataFrame
    y: pd.Series
    try:
        df.dropna()
        X = df.drop(columns=[target_column, "flight_number_operating_airline"]).copy()
        y = df[target_column].copy()
        print( "dropped; Splitting...")
    except Exception as e:
        print( f"dropping exception is: {str(e)}")
        print("Error occurred while dropping columns")
        print( df.info())
        print( df.describe()) 

    preprocessor = ColumnTransformer(transformers=[
        ('high_card_cat', TargetEncoder(target_type='continuous'), cat_columns),
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

    return model, X, y

def TrainModel( model: Pipeline, X: pd.DataFrame, y: pd.Series):
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    #print("splitted")    
    #print(f"DF Describe:\n{xTrain.describe()}")
    #print(f"DF Info:\n{xTrain.info()}")     
    #now = datetime.now()
    #time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    #print(f"Fitting start: {time_string}")

    model.fit(xTrain, yTrain)
    #now = datetime.now()
    #time_string = now.strftime("%Y-%m-%d %H:%M:%S")

    predictions = model.predict(xTest)
    #print("predictions done")
    mae = mean_absolute_error(yTest, predictions)
    r2 = r2_score(yTest, predictions)
    #print(f"Mean Absolute Error: {mae}")
    #print(f"R2 Score: {r2}")    
    return model, mae, r2

def SaveModel(model: Pipeline, path: str = f"./{MODELS_DIR}/{MODEL_NAME}"):
    print(f"Saving model to {MODEL_NAME} using pickle...")
    Path(f"./{MODELS_DIR}").mkdir(parents=True, exist_ok=True)

    with open(f"./{MODELS_DIR}/{MODEL_NAME}", 'wb') as file:    
        pickle.dump(model, file)

    print(f"Model saved successfully with pickle to {MODELS_DIR}/{MODEL_NAME}")

if __name__ == "__main__":
    df = LoadData()
    cat_columns = [ "origin", "dest", "origin_airport_id", "dest_airport_id", 
                    "airline", "operating_airline", "flight_number_marketing_airline", 
                    "tail_number", 
                  ]
    numeric_columns = ["year", "month", "day_of_month", "day_of_week",
        'crs_dep_hrs', 'crs_dep_mins', 'crs_arr_hrs', 'crs_arr_mins', "distance"] 
    target_column = "arr_delay"

    model, X, y = CreateModel(df,categorical_columns=cat_columns, numeric_columns=numeric_columns, target_column=target_column)    
    model, mae, r2 = TrainModel(model, X, y)
    print(f"Model trained with MAE: {mae} and R2: {r2}")
    SaveModel(model=model)

