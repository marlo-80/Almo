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
    df = shuffle_dataset(df,[0.5,0.5,0.5], max_rows=250_000)
    print("Dataset shuffled")
    return df

def TrainModel():
    df = LoadData()
    print(f"Dataset loaded with shape: {df.shape}")    

    target_column = "ArrDelay"
    df.dropna(subset=target_column, inplace=True)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print("Split done")    
    print(f"DF Describe:\n{xTrain.describe()}")
    print(f"DF Info:\n{xTrain.info()}")
    
    target_columns = [  "Origin", "Dest", "OriginAirportID", "DestAirportID", #"OriginCityName", "DestCityName",
                        "Airline", "Operating_Airline",
                        "Flight_Number_Marketing_Airline", "Tail_Number", 
                     ]
    numeric_columns = ["Year", "Month", "DayofMonth", 'DayOfWeek','CRSDeptHrs', 'CRSDepMins', 'CRSArrHrs', 'CRSArrMins', "Distance"] 

    preprocessor = ColumnTransformer(transformers=[
        ('high_card_cat', TargetEncoder(target_type='continuous'), target_columns),
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

    predictions = model.predict(xTest)
    print("predictions done")
    mae = mean_absolute_error(yTest, predictions)
    r2 = r2_score(yTest, predictions)
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    print(f"Saving model to {MODEL_NAME} using pickle...")
    Path(f"./{MODELS_DIR}").mkdir(parents=True, exist_ok=True)

    with open(f"./{MODELS_DIR}/{MODEL_NAME}", 'wb') as file:    
        pickle.dump(model, file)

    print(f"Model saved successfully with pickle to {MODELS_DIR}/{MODEL_NAME}")

if __name__ == "__main__":
    TrainModel()
    

