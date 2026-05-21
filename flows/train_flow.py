# flows/train_flow.py
import pandas as pd
from prefect import flow, task
from src.data import load_subset_table
from src.preprocessing import build_preprocessor
from src.train import train_and_log
from sklearn.ensemble import RandomForestRegressor

from flows.config import DEFAULT_CONFIG
from src.data import load_subset_table, convert_integers_to_float


@task
def load_and_clean_data(query: str, numeric_cols: list[str]) -> pd.DataFrame:
    df = load_subset_table(query)
    df = convert_integers_to_float(df, numeric_cols)
    return df

@task
def split_data(df: pd.DataFrame, target: str) -> tuple:
    """Chronologischer Split (Tabelle ist bereits nach flight_date sortiert)."""
    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]
    return train, val, test

@task
def build_model(model_type: str, model_params: dict):
    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(**model_params)
    raise ValueError(f"Unbekannter model_type: {model_type}")

@task
def run_training(train_df, val_df, config: dict):
    # Preprocessor und Modell erstellen
    preprocessor = build_preprocessor(
        numeric_cols=config["numeric_cols"],
        categorical_cols=config["categorical_cols"],
        impute_num=config.get("impute_num", "median"),
        impute_cat=config.get("impute_cat", "most_frequent"),
    )
    model = build_model.fn(config["model_type"], config["model_params"])

    pipeline = train_and_log(train_df, val_df, preprocessor, model, config)
    return pipeline

@flow(name="flight-delay-training")
def training_pipeline(config: dict = DEFAULT_CONFIG):
    df = load_and_clean_data(config["dataset_query"], config["numeric_cols"])
    train, val, test = split_data(df, config["target"])
    pipeline = run_training(train, val, config)
    return pipeline

if __name__ == "__main__":
    import sys
    from flows.config import DEFAULT_CONFIG

    config_name = sys.argv[1] if len(sys.argv) > 1 else "DEFAULT_CONFIG"
    import flows.config as cfg_module
    config = getattr(cfg_module, config_name, DEFAULT_CONFIG)
    training_pipeline(config)