# src/preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import re
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnNameNormalizer(BaseEstimator, TransformerMixin):
    """Wandelt alle Spaltennamen in snake_case um."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = [
            re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower().replace(' ', '_')
            for col in X.columns
        ]
        return X


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    impute_num: str = "median",
    impute_cat: str = "most_frequent",
) -> ColumnTransformer:
    """Baut einen ColumnTransformer mit Imputation und Skalierung/Encoding."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_cat)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    return preprocessor