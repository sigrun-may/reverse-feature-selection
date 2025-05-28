# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Main tests script for reverse feature selection with random forest."""

import numpy as np
import pandas as pd
import pytest
from mltb2.data import load_colon

from reverse_feature_selection.reverse_random_forests import (
    calculate_oob_errors,
    remove_features_correlated_to_target_feature,
    select_feature_subset,
)


def load_test_data():
    """Load tests data."""
    y, x = load_colon()
    train_df = pd.DataFrame(x)
    train_df.insert(loc=0, column="label", value=y)
    train_df.columns = train_df.columns.astype(str)

    # reset index for cross validation splits
    data_df = train_df.reset_index(drop=True)
    assert data_df.columns[0] == "label"
    return train_df


def test_calculate_oob_errors_with_valid_input():
    """Test calculating out-of-bag errors for a valid input."""
    data_df = load_test_data().iloc[:, :50]
    seeds = [0, 1]
    meta_data = {"n_cpus": 2, "random_seeds": seeds, "train_correlation_threshold": 0.2}
    labeled_errors, unlabeled_errors, number_of_features_in_training_data = calculate_oob_errors(
        "1", data_df, data_df.corr(method="spearman"), meta_data
    )
    assert labeled_errors is not None
    assert unlabeled_errors is not None
    assert number_of_features_in_training_data > 0
    assert isinstance(labeled_errors, list)
    assert isinstance(unlabeled_errors, list)
    assert len(labeled_errors) == len(seeds) == len(unlabeled_errors)


def test_calculate_oob_errors_with_invalid_target_feature():
    """Test calculating out-of-bag errors for an invalid target feature."""
    data_df = load_test_data().iloc[:, :50]
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(KeyError):
        calculate_oob_errors("invalid_feature", data_df, data_df.corr(), meta_data)


def test_calculate_feature_importance_with_invalid_train_indices():
    """Test calculating feature importance with invalid train indices."""
    data_df = pd.DataFrame(
        {"label": [1, 0, 1, 0, 1], "feature1": [0.1, 0.2, 0.3, 0.4, 0.5], "feature2": [0.5, 0.4, 0.3, 0.2, 0.1]}
    )
    train_indices = np.array([0, 1, 5000000])  # index 5000000 is out of bounds
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(IndexError):
        select_feature_subset(data_df, train_indices, meta_data=meta_data)


def test_calculate_feature_importance_with_no_features():
    """Test calculating feature importance with no features."""
    data_df = pd.DataFrame({"label": [1, 0, 1, 0, 1]})
    train_indices = np.array([0, 1, 2])
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(ValueError):
        select_feature_subset(data_df, train_indices, meta_data=meta_data)


def test_calculate_oob_errors_with_correlated_features_only():
    """Test calculating out-of-bag errors with correlated features only."""
    data_df = pd.DataFrame(
        {"label": [1, 0, 1, 0, 1], "feature1": [0.1, 0.2, 0.3, 0.4, 0.5], "feature2": [1, 2, 3, 4, 5]}
    )
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(AssertionError):
        calculate_oob_errors("feature1", data_df, data_df.corr(method="spearman"), meta_data)


def test_removal_of_correlated_features():
    """Test removal of correlated features."""
    from mltb2.data import load_colon

    y, x = load_colon()
    train_df = pd.DataFrame(x)
    train_df.insert(loc=0, column="label", value=y)
    train_df.columns = train_df.columns.astype(str)
    correlation_matrix_df = train_df.corr(method="spearman")
    target_feature = "1"
    meta_data = {"train_correlation_threshold": 0.2}

    uncorrelated_train_df = remove_features_correlated_to_target_feature(
        train_df, correlation_matrix_df, target_feature, meta_data
    )
    assert uncorrelated_train_df.columns[0] == "label"
    assert target_feature not in uncorrelated_train_df.columns
    uncorrelated_features = uncorrelated_train_df.columns[1:]

    # Check if the maximum correlation of any uncorrelated feature with the target is within the threshold
    assert (
        correlation_matrix_df.loc[target_feature, uncorrelated_features].abs().max()
        <= meta_data["train_correlation_threshold"]
    ), f"{meta_data['train_correlation_threshold']}"

    # Check if all correlation values left, which are not included in the uncorrelated_features are above the threshold
    correlated_features = correlation_matrix_df.columns[
        correlation_matrix_df.loc[target_feature].abs() > meta_data["train_correlation_threshold"]
    ]
    assert (
        correlation_matrix_df.loc[target_feature, correlated_features].abs().min()
        > meta_data["train_correlation_threshold"]
    ), f"{meta_data['train_correlation_threshold']}"


def test_removal_of_correlated_features_no_features_uncorrelated_to_target_feature():
    """Test removal of correlated features when no features are uncorrelated to the target feature."""
    train_df = pd.DataFrame({"label": [1, 0, 1], "feature1": [10, 20, 30], "feature2": [100, 200, 300]})
    correlation_matrix_df = train_df.corr(method="spearman")
    target_feature = "feature1"
    meta_data = {"train_correlation_threshold": 0.6}

    with pytest.raises(AssertionError):
        remove_features_correlated_to_target_feature(train_df, correlation_matrix_df, target_feature, meta_data)
