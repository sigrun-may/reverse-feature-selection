# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Main script for cacluating the raw results of the feature selection."""

import multiprocessing
import pickle
import random
import sys
from pathlib import Path

import cross_validation
import git

# from standard_rf import optimize as calculate_feature_importance
from data_loader_tools import load_data_with_standardized_sample_size
from standard_rf import optimize as calculate_feature_importance
from reverse_feature_selection.reverse_rf_random import select_feature_subset


def main():
    """Calculate the raw results of the feature selection."""
    # parse result path from input
    result_base_path = Path(sys.argv[1])
    print("result data_path: ", result_base_path)

    # save git commit hash
    git_repository = git.Repo(search_parent_directories=True)
    commit_sha = git_repository.head.object.hexsha

    # define meta data for the experiment
    meta_data_dict = {
        "experiment_id": "deleteleukemia_shift_00",
        # valid data names for the data loader are "colon", "prostate" or "leukemia_big"
        "data_name": "colon",
        "description": "Leukemia dataset",
        "git_commit_hash": commit_sha,
        "n_cpus": multiprocessing.cpu_count(),  # number of available CPUs
        "train_correlation_threshold": 0.2,
        # generate random seeds for reproducibility of reverse random forest
        "random_seeds": [
            29,
            10,
            17,
            42,
            234,
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
            23,
            34,
            45,
            56,
            67,
            78,
            89,
            90,
        ],
        # generate random seed for reproducibility of random forest
        "random_state": random.randint(1, 2**8),
        "verbose_optuna": True,
        "n_trials_optuna": 50,
    }

    # # shift seeds by 10
    # meta_data_dict["random_seeds"] = [seed + 10 for seed in meta_data_dict["random_seeds"]]

    # # shift seeds by 20
    # meta_data_dict["random_seeds"] = [seed + 20 for seed in meta_data_dict["random_seeds"]]

    # load previous result for the given experiment, if available
    result_dict_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_result_dict.pkl")
    if result_dict_path.exists():
        with open(result_dict_path, "rb") as file:
            result_dict = pickle.load(file)
    else:
        result_dict = {}

    # load data
    data_df = load_data_with_standardized_sample_size(meta_data_dict["data_name"])

    # # load artificial data for testing from csv file
    # import pandas as pd
    # data_df = pd.read_csv(
    #     "/home/sigrun/PycharmProjects/reverse_feature_selection/data/artificial_biomarker_data_2.csv"
    # ).iloc[:, :70]
    # del result_dict["standard_random_forest"]
    # assert "standard_random_forest" not in result_dict, "Standard random forest results are still present."

    print("number of samples", data_df.shape[0], "number of features", data_df.shape[1] - 1)

    # # calculate raw feature subset data for reverse random forest
    # result_dict["reverse_random_forest"] = cross_validation.cross_validate(
    #     data_df, meta_data_dict, select_feature_subset
    # )
    # result_dict["reverse_random_forest_meta_data"] = meta_data_dict

    # calculate raw feature subset data for standard random forest
    result_dict["standard_random_forest"] = cross_validation.cross_validate(
        data_df, meta_data_dict, calculate_feature_importance
    )
    result_dict["standard_random_forest_meta_data"] = meta_data_dict

    # save results
    result_base_path.mkdir(parents=True, exist_ok=True)
    result_dict_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_result_dict.pkl")
    with open(result_dict_path, "wb") as file:
        pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
