# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Main test script for cacluating a grid of repeated feature selection experiments for reverse feature selection."""


from feature_selection_benchmark.repeated_experiments_reverse_rf import define_random_seeds


def test_define_random_seeds():
    """Test defining random seeds for reproducibility of the experiments."""
    seeds = define_random_seeds()
    assert len(seeds) == 3
    assert len(seeds[0]) == 30
    assert len(seeds[1]) == 30
    assert len(seeds[2]) == 30
    flattened_seeds = [seed for sublist in seeds for seed in sublist]
    assert len(flattened_seeds) == 90
    assert len(set(flattened_seeds)) == 90
