# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data loader tools test script."""

import unittest
from unittest.mock import patch
import os

import numpy as np
import pandas as pd

from feature_selection_benchmark.data_loader_tools import (
    convert_to_single_df,
    get_indices_for_class,
    load_data_df,
    load_train_test_data_for_standardized_sample_size,
    shuffle,
    standardize_sample_size_of_hold_out_data,
)


class TestDataLoaderTools(unittest.TestCase):
    """Test data loader tools."""

    def tearDown(self):
        """Clean up and delete the generated file."""
        if os.path.exists("./test_experiment_id_data_df.csv"):
            os.remove("./test_experiment_id_data_df.csv")

    def test_standardize_sample_size_of_hold_out_data(self):
        """Test standardizing the sample size of the hold out data."""
        data = pd.DataFrame(np.random.randn(20, 2000))  # 20 samples, 2000 features
        label = pd.Series([0] * 10 + [1] * 10)
        hold_out_data_df = convert_to_single_df(data, label)
        balanced_df = standardize_sample_size_of_hold_out_data(hold_out_data_df, shuffle_seed=42)
        self.assertEqual(balanced_df["label"].value_counts().tolist(), [7, 7])

    def test_convert_to_single_df(self):
        """Test converting data and labels into a single dataframe."""
        x = pd.DataFrame(np.random.randn(30, 2000))
        y = pd.Series([0] * 15 + [1] * 15)
        df = convert_to_single_df(x, y)
        self.assertEqual(df.shape, (30, 2001))
        self.assertEqual(df.columns[0], "label")
        self.assertTrue(all(df["label"].isin([0, 1])))

    def test_get_indices_for_class(self):
        """Test getting train and test indices for a specific class label."""
        label = pd.Series([0] * 20 + [1] * 20)
        train_indices, test_indices = get_indices_for_class(label, 0, 15)
        self.assertEqual(len(train_indices), 15)
        self.assertEqual(len(test_indices), 5)

    def test_shuffle(self):
        """Test shuffling data and labels."""
        data = pd.DataFrame(np.random.randn(40, 2000))
        label = pd.Series([0] * 20 + [1] * 20)
        shuffled_data, shuffled_label = shuffle(data, label, shuffle_seed=42)
        self.assertEqual(shuffled_data.shape, data.shape)
        self.assertEqual(shuffled_label.shape, label.shape)
        self.assertFalse(shuffled_data.equals(data))

    @patch("feature_selection_benchmark.data_loader_tools.load_colon")
    def test_load_train_test_data_for_standardized_sample_size(self, mock_load_colon):
        """Test loading train and test data with standardized sample size."""
        mock_load_colon.return_value = (pd.Series([0] * 30 + [1] * 25), pd.DataFrame(np.random.randn(55, 2000)))
        meta_data_dict = {"data_name": "colon", "shuffle_seed": 42}
        train_df, test_df = load_train_test_data_for_standardized_sample_size(meta_data_dict)
        self.assertEqual(train_df.shape, (30, 2001))
        self.assertEqual(test_df.shape, (25, 2001))
        self.assertEqual(train_df["label"].value_counts().tolist(), [15, 15])
        self.assertEqual(test_df["label"].value_counts().tolist(), [15, 10])

    @patch("feature_selection_benchmark.data_loader_tools.load_colon")
    @patch("pandas.read_csv")
    def test_load_data_df(self, mock_read_csv, mock_load_colon):
        """Test loading data as a pandas DataFrame."""
        mock_load_colon.return_value = (pd.Series([0] * 30 + [1] * 25), pd.DataFrame(np.random.randn(55, 2000)))
        mock_read_csv.return_value = pd.DataFrame(np.random.randn(30, 2001))

        meta_data_dict = {"data_name": "colon", "experiment_id": "test", "shuffle_seed": 42}
        df = load_data_df(meta_data_dict)
        self.assertEqual(df.shape, (30, 2001))

        meta_data_dict = {"data_name": "random", "experiment_id": "test", "shuffle_seed": 42}
        df = load_data_df(meta_data_dict)
        self.assertEqual(df.shape, (30, 2001))

        meta_data_dict = {"data_name": "colon", "experiment_id": "test_experiment_id", "shuffle_seed": None}
        df = load_data_df(meta_data_dict, generate_random_noise=True, path_for_random_noise=".")
        self.assertEqual(df.shape, (30, 2001))


if __name__ == "__main__":
    unittest.main()

