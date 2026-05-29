# src/preprocessing.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# Hilfs-Transformer
# ---------------------------------------------------------------------------

def _sin_cos_transformer(X: np.ndarray) -> np.ndarray:
    """Wandelt Uhrzeiten im Format HHMM (integer) in sin/cos-Komponenten um."""
    X = np.atleast_2d(X)
    hours = X // 100
    minutes = X % 100
    total_minutes = hours * 60 + minutes
    radians = 2.0 * np.pi * total_minutes / 1440.0
    return np.column_stack([np.sin(radians), np.cos(radians)])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Ersetzt jeden Kategoriewert durch seine relative Häufigkeit im Trainings-Set."""

    def __init__(self):
        self.mappings = {}

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mappings = {}
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            unique, counts = np.unique(col.astype(str), return_counts=True)
            total = len(col)
            freq_map = {k: v / total for k, v in zip(unique, counts)}
            self.mappings[col_idx] = freq_map
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_out = np.zeros_like(X, dtype=float)
        for col_idx in range(X.shape[1]):
            freq_map = self.mappings[col_idx]
            col_str = X[:, col_idx].astype(str)
            for i, val in enumerate(col_str):
                X_out[i, col_idx] = freq_map.get(val, 0.0)
        return X_out


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def build_preprocessor(
    low_card_cols: list[str],
    high_card_cols: list[str],
    cyclic_cols: list[str],
    numeric_cols: list[str],
    skewed_numeric_cols: list[str],
    low_card_strategy: str = "onehot",      # "onehot" | "ordinal"
    high_card_strategy: str = "target",     # "target" | "ordinal" | "frequency"
    impute_num: str = "median",
    impute_cat: str = "most_frequent",
    target_type: str = "continuous",        # "continuous" | "binary"
) -> ColumnTransformer:
    """
    Baut den vollständigen ColumnTransformer.

    Strategien:
      - low_card_strategy:  onehot / ordinal
      - high_card_strategy: target / ordinal / frequency
    """

    # ---- Low‑Cardinality --------------------------------------------------
    if low_card_strategy == "onehot":
        low_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_cat)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    elif low_card_strategy == "ordinal":
        low_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_cat)),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    else:
        raise ValueError(f"Unbekannte low_card_strategy: {low_card_strategy}")

    # ---- High‑Cardinality -------------------------------------------------
    if high_card_strategy == "target":
        high_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_cat)),
            ("encoder", TargetEncoder(target_type=target_type)),
        ])
    elif high_card_strategy == "ordinal":
        high_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_cat)),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    elif high_card_strategy == "frequency":
        high_card_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=impute_cat)),
            ("encoder", FrequencyEncoder()),
        ])
    else:
        raise ValueError(f"Unbekannte high_card_strategy: {high_card_strategy}")

    # ---- Zyklische Uhrzeiten ----------------------------------------------
    cyclic_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("sin_cos", FunctionTransformer(_sin_cos_transformer, validate=False)),
    ])

    # ---- Standard‑Numerisch -----------------------------------------------
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("scaler", StandardScaler()),
    ])

    # ---- Schiefe Numerisch (Log + Scale) ----------------------------------
    skewed_num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
    ])

    # ---- Zusammenbau ------------------------------------------------------
    preprocessor = ColumnTransformer([
        ("low_card",    low_card_pipe,      low_card_cols),
        ("high_card",   high_card_pipe,     high_card_cols),
        ("cyclic",      cyclic_pipe,        cyclic_cols),
        ("num",         num_pipe,           numeric_cols),
        ("skewed_num",  skewed_num_pipe,    skewed_numeric_cols),
    ])

    return preprocessor