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

from data_loader_tools import load_data_with_standardized_sample_size

# from reverse_feature_selection.reverse_rf_random import select_feature_subset
# from standard_rf import optimize as calculate_feature_importance


def main():
    """Calculate the raw results of the feature selection."""
    # parse result path from input
    result_base_path = Path(sys.argv[1])
    print("result data_path: ", result_base_path)
    result_base_path.mkdir(parents=True, exist_ok=True)

    # save git commit hash
    git_repository = git.Repo(search_parent_directories=True)
    commit_sha = git_repository.head.object.hexsha

    # define meta data for the experiment
    meta_data_dict = {
        # "experiment_id": "leukemia_shift_02",
        # # valid data names for the data loader are "colon", "prostate" or "leukemia_big"
        # "data_name": "leukemia_big",
        # "description": "Leukemia dataset",
        "experiment_id": "random_noise_lognormal_30_7000_01",
        "data_name": "random_noise_lognormal_30_7000_01",
        "description": "Random noise dataset for testing purposes with 30 samples and 2000 features.",
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

    # shift seeds by 20
    meta_data_dict["random_seeds"] = [seed + 20 for seed in meta_data_dict["random_seeds"]]

    # load previous result for the given experiment, if available
    result_dict_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_result_dict.pkl")
    if result_dict_path.exists():
        with open(result_dict_path, "rb") as file:
            result_dict = pickle.load(file)
    else:
        result_dict = {}

    # # load data
    # data_df = load_data_with_standardized_sample_size(meta_data_dict["data_name"])

    # generate random noise data for benchmarking
    # import numpy as np
    # import pandas as pd
    # rnd = np.random.default_rng(seed=42)
    # data_df = pd.DataFrame(rnd.normal(scale=2, size=(30, 2000)), columns=[f"f_{i}" for i in range(2000)])
    # # generate array as binary balanced target with two classes for 30 samples
    # label_array = np.concatenate([np.zeros(15), np.ones(15)])
    # data_df.insert(0, "label", label_array)
    # # save data_df as csv
    # data_df_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_data_df.csv")
    # data_df.to_csv(data_df_path, index=False)

    # # load artificial data for testing from csv file
    import pandas as pd
    data_df = pd.read_csv(f"../data/{meta_data_dict['experiment_id']}_data_df.csv")
    # del result_dict["standard_random_forest"]
    # assert "standard_random_forest" not in result_dict, "Standard random forest results are still present."

    print("number of samples", data_df.shape[0], "number of features", data_df.shape[1] - 1)

    # # calculate raw feature subset data for reverse random forest
    # from reverse_feature_selection.reverse_rf_random import select_feature_subset
    #
    # result_dict["reverse_random_forest"] = cross_validation.cross_validate(
    #     data_df, meta_data_dict, select_feature_subset
    # )
    # result_dict["reverse_random_forest_meta_data"] = meta_data_dict

    # calculate raw feature subset data for standard random forest
    from standard_rf import calculate_feature_importance
    result_dict["standard_random_forest"] = cross_validation.cross_validate(
        data_df, meta_data_dict, calculate_feature_importance
    )
    result_dict["standard_random_forest_meta_data"] = meta_data_dict

    # save results
    result_dict_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_result_dict.pkl")
    with open(result_dict_path, "wb") as file:
        pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
