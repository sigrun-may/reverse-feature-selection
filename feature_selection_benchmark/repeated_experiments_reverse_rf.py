# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Main script for cacluating a grid of repeated feature selection experiments for reverse feature selection."""

import multiprocessing
import pickle
import sys
from pathlib import Path

import git

from feature_selection_benchmark import cross_validation
from feature_selection_benchmark.data_loader_tools import load_data_df
from reverse_feature_selection.reverse_rf_random import select_feature_subset


def define_random_seeds() -> list:
    """Define random seeds for reproducibility of the experiments.

    Returns:
        A list of three lists with 30 different random seeds each.

    Raises:
        ValueError: If the random seeds are not unique.
    """
    # define a set of 30 different random seeds for reproducibility of the experiments
    random_seeds_0 = [
        29,
        10,
        17,
        42,
        213,
        34,
        1,
        5,
        19,
        3,
        23,
        9,
        7,
        123,
        234,
        345,
        456,
        567,
        678,
        789,
        890,
        15,
        333,
        37,
        45,
        56,
        67,
        78,
        89,
        90,
    ]
    # generate a set of 30 different random seeds for reproducibility of reverse random forest
    # distinct to the seeds in random_seeds_0
    random_seeds_1 = [seed + 97 for seed in random_seeds_0]
    random_seeds_2 = [seed + 1948 for seed in random_seeds_0]

    list_of_seeds = [random_seeds_0, random_seeds_1, random_seeds_2]

    # flatten list of seeds to check if the random seeds are equal
    flattened_lists_of_seeds = [seed for sublist in list_of_seeds for seed in sublist]
    assert len(flattened_lists_of_seeds) == 90, "Number of random seeds is not equal to 90."
    if len(set(flattened_lists_of_seeds)) != 90:
        # find equal elements within the list
        equal_elements = [
            element for element in flattened_lists_of_seeds if flattened_lists_of_seeds.count(element) > 1
        ]
        print("number of equal elements: ", len(set(equal_elements)), set(equal_elements))
        raise ValueError("Random seeds are not unique.")

    return list_of_seeds


def main():
    """Main function for calculating a grid of repeated feature selection experiments for reverse feature selection."""
    # parse result path from input
    result_base_path = Path(sys.argv[1])
    print("result data_path: ", result_base_path)
    result_base_path.mkdir(parents=True, exist_ok=True)

    # seed to shuffle the indices of the samples of the data set
    shuffle_seed = None

    # valid data names for the data loader are "colon", "prostate" or "leukemia_big"
    # data_names = ["colon", "prostate", "leukemia_big"]
    data_names = ["random_noise_lognormal", "random_noise_normal"]
    for data_name in data_names:
        # repeat the experiment three times with different random seeds
        for i, list_of_random_seeds in enumerate(define_random_seeds()):
            experiment_id = f"{data_name}_{i}_shuffle_seed_{shuffle_seed}"

            # define meta data for the experiment
            meta_data_dict = {
                "git_commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
                "experiment_id": experiment_id,
                "data_name": data_name,
                "n_cpus": multiprocessing.cpu_count(),  # number of available CPUs
                "train_correlation_threshold": 0.2,
                # random seeds for reproducibility of reverse random forest
                "random_seeds": list_of_random_seeds,
                "shuffle_seed": shuffle_seed,
            }
            if "random" in data_name:
                meta_data_dict["description"] = "Random noise dataset for testing purposes."
                meta_data_dict["data_shape_random_noise"] = (62, 2000)
                # The path to the directory where generated random noise is stored.
                meta_data_dict["path_for_random_noise"] = f"{result_base_path}/random_noise"
            else:
                meta_data_dict["description"] = f"{data_name} dataset."

            # load data
            data_df = load_data_df(meta_data_dict)
            print("number of samples", data_df.shape[0], "number of features", data_df.shape[1] - 1)

            # calculate raw feature subset data for reverse random forest
            result_dict = {
                "reverse_random_forest": cross_validation.cross_validate(
                    data_df, meta_data_dict, select_feature_subset
                ),
                "reverse_random_forest_meta_data": meta_data_dict,
            }

            # save results
            result_dict_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_result_dict.pkl")
            with open(result_dict_path, "wb") as file:
                pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
