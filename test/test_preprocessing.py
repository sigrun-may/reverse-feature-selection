# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Preprocessing test script."""

import pandas as pd
import pytest

from reverse_feature_selection.preprocessing import remove_features_correlated_to_target_feature


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
