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


# ============================================================================
# Helper Transformers
# ============================================================================

def _sin_cos_transformer(X: np.ndarray) -> np.ndarray:
    """
    Convert time values in HHMM format to sine/cosine components.

    This transformer is used for cyclic encoding of time features (e.g.,
    departure/arrival times). The time is converted to minutes since midnight,
    then normalized to a 24‑hour cycle using sine and cosine transformations
    to preserve the cyclic nature of time (e.g., 23:59 and 00:00 are close).

    Args:
        X (np.ndarray): Input array containing integer HHMM values (e.g., 900 = 09:00).

    Returns:
        np.ndarray: Array with two columns per input column:
                    [sin(angle), cos(angle)] where angle ∈ [0, 2π).
    """
    X = np.atleast_2d(X)
    hours = X // 100
    minutes = X % 100
    total_minutes = hours * 60 + minutes
    radians = 2.0 * np.pi * total_minutes / 1440.0
    return np.column_stack([np.sin(radians), np.cos(radians)])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Custom encoder that replaces categorical values with their relative frequency.

    This transformer is useful for high‑cardinality categorical features where
    one‑hot encoding would be too sparse. Each value is replaced by its
    proportion (frequency) in the training set. Unknown values in the test set
    are mapped to 0.0.

    Example:
        >>> X = np.array([['A'], ['B'], ['A'], ['C']])
        >>> encoder = FrequencyEncoder().fit(X)
        >>> encoder.transform(X)
        array([[0.5], [0.25], [0.5], [0.25]])

    Attributes:
        mappings (dict): For each column index, a dict mapping category value
                         to its relative frequency in the training set.
    """
    def __init__(self):
        """
        Initialize the frequency encoder with an empty mappings dictionary.
        """
        self.mappings = {}

    def fit(self, X, y=None):
        """
        Learn the frequency mapping for each column from the training data.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).
            y (optional): Ignored. Provided for scikit‑learn API compatibility.

        Returns:
            FrequencyEncoder: The fitted encoder instance.
        """
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
        """
        Transform the input data by replacing values with their frequencies.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed array with frequency values (float64).
        """
        X = np.asarray(X)
        X_out = np.zeros_like(X, dtype=float)
        for col_idx in range(X.shape[1]):
            freq_map = self.mappings[col_idx]
            col_str = X[:, col_idx].astype(str)
            for i, val in enumerate(col_str):
                X_out[i, col_idx] = freq_map.get(val, 0.0)
        return X_out


# ============================================================================
# Main Preprocessor Builder
# ============================================================================

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
    Build a complete preprocessing pipeline as a scikit‑learn ColumnTransformer.

    The pipeline handles five groups of features:
    1. Low‑cardinality categorical features (few unique values)
       → One‑hot or ordinal encoding.
    2. High‑cardinality categorical features (many unique values)
       → Target, ordinal, or frequency encoding.
    3. Cyclic time features (e.g., CRSDepTime, CRSArrTime)
       → Sin/cos transformation.
    4. Standard numeric features
       → Imputation + StandardScaler.
    5. Skewed numeric features
       → Imputation + log transform + StandardScaler.

    Args:
        low_card_cols (list[str]): Columns with few unique values (e.g., month, day_of_week).
        high_card_cols (list[str]): Columns with many unique values (e.g., airport IDs).
        cyclic_cols (list[str]): Time columns in HHMM format requiring cyclic encoding.
        numeric_cols (list[str]): Standard numeric features (e.g., crs_elapsed_time).
        skewed_numeric_cols (list[str]): Numeric features with skewed distribution (e.g., distance).
        low_card_strategy (str): Encoding for low‑cardinality features.
                                 Options: "onehot" (default) or "ordinal".
        high_card_strategy (str): Encoding for high‑cardinality features.
                                  Options: "target" (default), "ordinal", or "frequency".
        impute_num (str): Imputation strategy for numeric columns.
                          Options: "median" (default), "mean", "most_frequent".
        impute_cat (str): Imputation strategy for categorical columns.
                          Options: "most_frequent" (default).
        target_type (str): Type of target variable for TargetEncoder.
                           Options: "continuous" (default) or "binary".

    Returns:
        ColumnTransformer: A fully configured scikit‑learn preprocessor.

    Raises:
        ValueError: If an invalid `low_card_strategy` or `high_card_strategy` is provided.

    Example:
        >>> preprocessor = build_preprocessor(
        ...     low_card_cols=["month", "day_of_week"],
        ...     high_card_cols=["origin_airport_id"],
        ...     cyclic_cols=["crs_dep_time"],
        ...     numeric_cols=["crs_elapsed_time"],
        ...     skewed_numeric_cols=["distance"],
        ...     low_card_strategy="onehot",
        ...     high_card_strategy="target"
        ... )
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

    # ---- Cyclic Times -----------------------------------------------------
    cyclic_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("sin_cos", FunctionTransformer(_sin_cos_transformer, validate=False)),
    ])

    # ---- Numeric default --------------------------------------------------
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("scaler", StandardScaler()),
    ])

    # ---- Skewed numeric (log + scale) -------------------------------------
    skewed_num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_num)),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
    ])

    # ---- Assemly ----------------------------------------------------------
    preprocessor = ColumnTransformer([
        ("low_card",    low_card_pipe,      low_card_cols),
        ("high_card",   high_card_pipe,     high_card_cols),
        ("cyclic",      cyclic_pipe,        cyclic_cols),
        ("num",         num_pipe,           numeric_cols),
        ("skewed_num",  skewed_num_pipe,    skewed_numeric_cols),
    ])

    return preprocessor