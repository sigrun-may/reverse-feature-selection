import numpy as np
import pandas as pd
import pytest
from mltb2.data import load_colon

from reverse_feature_selection.reverse_rf_random import calculate_oob_errors, select_feature_subset


def load_test_data():
    y, x = load_colon()
    train_df = pd.DataFrame(x)
    train_df.insert(loc=0, column="label", value=y)
    train_df.columns = train_df.columns.astype(str)

    # reset index for cross validation splits
    data_df = train_df.reset_index(drop=True)
    assert data_df.columns[0] == "label"
    return train_df


def test_calculate_oob_errors_with_valid_input():
    data_df = load_test_data().iloc[:, :50]
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    labeled_errors, unlabeled_errors = calculate_oob_errors("1", data_df, data_df.corr(method="spearman"), meta_data)
    assert labeled_errors is not None
    assert unlabeled_errors is not None


def test_calculate_oob_errors_with_invalid_target_feature():
    data_df = load_test_data().iloc[:, :50]
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(KeyError):
        calculate_oob_errors("invalid_feature", data_df, data_df.corr(), meta_data)


def test_calculate_feature_importance_with_invalid_train_indices():
    data_df = pd.DataFrame(
        {"label": [1, 0, 1, 0, 1], "feature1": [0.1, 0.2, 0.3, 0.4, 0.5], "feature2": [0.5, 0.4, 0.3, 0.2, 0.1]}
    )
    train_indices = np.array([0, 1, 5000000])  # index 5000000 is out of bounds
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(IndexError):
        select_feature_subset(data_df, train_indices, meta_data)


def test_calculate_feature_importance_with_no_features():
    data_df = pd.DataFrame({"label": [1, 0, 1, 0, 1]})
    train_indices = np.array([0, 1, 2])
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(ValueError):
        select_feature_subset(data_df, train_indices, meta_data)


def test_calculate_oob_errors_with_correlated_features_only():
    data_df = pd.DataFrame(
        {"label": [1, 0, 1, 0, 1], "feature1": [0.1, 0.2, 0.3, 0.4, 0.5], "feature2": [1, 2, 3, 4, 5]}
    )
    meta_data = {"n_cpus": 2, "random_seeds": [0, 1], "train_correlation_threshold": 0.2}
    with pytest.raises(ValueError):
        calculate_oob_errors("feature1", data_df, data_df.corr(method="spearman"), meta_data)
