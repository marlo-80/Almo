"""
Unit tests for the preprocessing pipeline builder.

This module tests the `build_preprocessor` function with a simple example
to ensure it can handle unseen categorical values without raising errors.
"""

import pytest
import pandas as pd
from src.preprocessing import build_preprocessor


def test_preprocessor_handles_unknown_categories():
    """
    Test that the preprocessor correctly handles unknown categories at transform time.

    The preprocessor is fitted on a small training set with known categories
    ('JFK', 'LAX', 'ORD') and numeric columns. During transform, an unseen
    category ('XYZ') should be encoded as zeros (or ignored) without raising
    an error. The output shape should match the expected number of features.

    This test uses:
        - low_cardinality_cols: ['origin']
        - numeric_cols: ['dep_delay']
        - low_card_strategy: 'onehot' (so 'XYZ' becomes all zeros)
        - high_card_strategy: (empty) – not used here
    """
    # Training data
    train_df = pd.DataFrame({
        'origin': ['JFK', 'LAX', 'ORD'],
        'dep_delay': [5, 10, 0]
    })

    # Build preprocessor with the correct current signature
    preprocessor = build_preprocessor(
        low_card_cols=['origin'],
        high_card_cols=[],          # no high-cardinality columns in this test
        cyclic_cols=[],             # no cyclic columns
        numeric_cols=['dep_delay'],
        skewed_numeric_cols=[],     # no skewed columns
        low_card_strategy='onehot',
        high_card_strategy='target',   # not used, but required
        impute_num='median',
        impute_cat='most_frequent',
        target_type='continuous'
    )

    preprocessor.fit(train_df)

    # Test with an unseen category
    test_df = pd.DataFrame({
        'origin': ['XYZ'],
        'dep_delay': [2]
    })

    transformed = preprocessor.transform(test_df)

    # The shape should be (1, number_of_features)
    # For one-hot encoding of 'origin' with 3 categories, we get 3 columns
    # plus 1 numeric column => 4 features total.
    assert transformed.shape == (1, 4)
    # The first three columns (one-hot) should be all zeros because 'XYZ' is unknown
    assert (transformed[0, :3] == 0).all()