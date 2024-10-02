# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data loader tools test script."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.data_loader_tools import (
    convert_to_single_df,
    get_indices_for_class,
    load_data_df,
    load_train_test_data_for_standardized_sample_size,
    shuffle,
    standardize_sample_size_of_hold_out_data,
    standardize_sample_size_of_hold_out_data_single_df,
)

# Create a random number generator
rng = np.random.default_rng()


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    if os.path.exists("./test_experiment_id_data_df.csv"):
        os.remove("./test_experiment_id_data_df.csv")
    if os.path.exists("./random_lognormal_test_(30, 2000)_df.csv"):
        os.remove("./random_lognormal_test_(30, 2000)_df.csv")
    if os.path.exists("./random_test_(30, 2000)_df.csv"):
        os.remove("./random_test_(30, 2000)_df.csv")


def test_standardize_sample_size_of_hold_out_data_single_df():
    data = pd.DataFrame(rng.standard_normal((20, 2000)))  # 20 samples, 2000 features
    label = pd.Series([0] * 10 + [1] * 10)
    hold_out_data_df = convert_to_single_df(data, label)
    balanced_df = standardize_sample_size_of_hold_out_data_single_df(hold_out_data_df, shuffle_seed=42)
    assert balanced_df["label"].value_counts().tolist() == [7, 7]


def test_standardize_sample_size_of_hold_out_data_single_df_no_shuffle_seed():
    data = pd.DataFrame(rng.standard_normal((20, 2000)))
    label = pd.Series([0] * 10 + [1] * 10)
    hold_out_data_df = convert_to_single_df(data, label)
    balanced_df = standardize_sample_size_of_hold_out_data_single_df(hold_out_data_df, shuffle_seed=None)
    assert balanced_df["label"].value_counts().tolist() == [7, 7]


def test_standardize_sample_size_of_hold_out_data_shuffled_indices():
    data = pd.DataFrame(rng.standard_normal((20, 2000)))
    label = pd.Series([0] * 10 + [1] * 10)
    hold_out_data_df = convert_to_single_df(data, label)
    start_balanced_df, start_label = standardize_sample_size_of_hold_out_data(hold_out_data_df, shuffle_seed=None)
    index_comparison = start_balanced_df.index
    np.testing.assert_array_equal(index_comparison, start_label.index)
    np.testing.assert_array_equal(index_comparison, start_balanced_df.index)
    for seed in range(10):
        balanced_df, balanced_label = standardize_sample_size_of_hold_out_data(hold_out_data_df, shuffle_seed=seed)
        assert set(balanced_df.index) != set(start_balanced_df.index)
        assert set(balanced_label.index) != set(start_label.index)
        assert set(balanced_df.index) != set(index_comparison)
        np.testing.assert_array_equal(balanced_df.index, balanced_label.index)
        index_comparison = balanced_df.index


def test_convert_to_single_df():
    x = pd.DataFrame(rng.standard_normal((30, 2000)))
    y = pd.Series([0] * 15 + [1] * 15)
    test_df = convert_to_single_df(x, y)
    assert test_df.shape == (30, 2001)
    assert test_df.columns[0] == "label"
    assert all(test_df["label"].isin([0, 1]))


def test_get_indices_for_class():
    label = pd.Series([0] * 20 + [1] * 20)
    train_indices, test_indices = get_indices_for_class(label, 0, 15)
    assert len(train_indices) == 15
    assert len(test_indices) == 5


def test_shuffle():
    data = pd.DataFrame(rng.standard_normal((40, 2000)))
    label = pd.Series([0] * 20 + [1] * 20)
    shuffled_data, shuffled_label = shuffle(data, label, shuffle_seed=42)
    assert shuffled_data.shape == data.shape
    assert shuffled_label.shape == label.shape
    assert not shuffled_data.equals(data)


@patch("feature_selection_benchmark.data_loader_tools.load_colon")
def test_load_train_test_data_for_standardized_sample_size(mock_load_colon):
    mock_load_colon.return_value = (pd.Series([0] * 30 + [1] * 25), pd.DataFrame(rng.standard_normal((55, 2000))))
    meta_data_dict = {"data_name": "colon", "shuffle_seed": 42}
    train_df, test_df = load_train_test_data_for_standardized_sample_size(meta_data_dict)
    assert train_df.shape == (30, 2001)
    assert test_df.shape == (25, 2001)
    assert train_df["label"].value_counts().tolist() == [15, 15]
    assert test_df["label"].value_counts().tolist() == [15, 10]


@patch("feature_selection_benchmark.data_loader_tools.load_colon")
@patch("pandas.read_csv")
def test_load_data_df(mock_read_csv, mock_load_colon):
    mock_load_colon.return_value = (pd.Series([0] * 30 + [1] * 25), pd.DataFrame(rng.standard_normal((55, 2000))))
    mock_read_csv.return_value = pd.DataFrame(rng.standard_normal((30, 2001)))

    meta_data_dict = {"data_name": "colon", "experiment_id": "test", "shuffle_seed": 42}
    test_df = load_data_df(meta_data_dict)
    assert test_df.shape == (30, 2001)

    meta_data_dict = {
        "data_name": "random_lognormal",
        "experiment_id": "random_lognormal_test",
        "shuffle_seed": 42,
        "path_for_random_noise": ".",
        "data_shape_random_noise": (30, 2000),
    }
    test_df = load_data_df(meta_data_dict)
    assert test_df.shape == (30, 2001)

    meta_data_dict = {
        "data_name": "random",
        "experiment_id": "random_test",
        "shuffle_seed": 42,
        "path_for_random_noise": ".",
        "data_shape_random_noise": (30, 2000),
    }
    test_df = load_data_df(meta_data_dict)
    assert test_df.shape == (30, 2001)

    meta_data_dict = {"data_name": "colon", "experiment_id": "test_experiment_id", "shuffle_seed": None}
    test_df = load_data_df(meta_data_dict)
    assert test_df.shape == (30, 2001)
