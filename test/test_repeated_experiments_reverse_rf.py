# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Main test script for cacluating a grid of repeated feature selection experiments for reverse feature selection."""

import unittest
from unittest.mock import MagicMock, patch

from feature_selection_benchmark.repeated_experiments_reverse_rf import define_random_seeds, main


class TestRepeatedExperimentsReverseRF(unittest.TestCase):
    """Test repeated experiments for reverse feature selection with random forest."""

    def test_define_random_seeds(self):
        """Test defining random seeds for reproducibility of the experiments."""
        seeds = define_random_seeds()
        self.assertEqual(len(seeds), 3)
        self.assertEqual(len(seeds[0]), 30)
        self.assertEqual(len(seeds[1]), 30)
        self.assertEqual(len(seeds[2]), 30)
        flattened_seeds = sum(seeds, [])
        self.assertEqual(len(flattened_seeds), 90)
        self.assertEqual(len(set(flattened_seeds)), 90)

    @patch("feature_selection_benchmark.repeated_experiments_reverse_rf.pickle.dump")
    @patch("feature_selection_benchmark.repeated_experiments_reverse_rf.cross_validation.cross_validate")
    @patch("feature_selection_benchmark.repeated_experiments_reverse_rf.load_data_df")
    @patch("feature_selection_benchmark.repeated_experiments_reverse_rf.Path.mkdir")
    @patch(
        "feature_selection_benchmark.repeated_experiments_reverse_rf.sys.argv",
        new=["script_name", "/home/sigrun/PycharmProjects/reverse_feature_selection/results"],
    )
    def test_main(self, mock_mkdir, mock_load_data_df, mock_cross_validate, mock_pickle_dump):
        """Test main function."""
        mock_load_data_df.return_value = MagicMock(shape=(30, 2001))
        mock_cross_validate.return_value = MagicMock()

        main()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_load_data_df.assert_called()
        mock_cross_validate.assert_called()
        mock_pickle_dump.assert_called()


if __name__ == "__main__":
    unittest.main()
