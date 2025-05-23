# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Main script for cacluating baseline feature selection experiments for standard random forest."""


import pickle
import sys
from pathlib import Path

import git
from ranger_rf import calculate_feature_importance

from feature_selection_benchmark import cross_validation
from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size


def analyze_file(file, result_base_path, path_to_random_noise_directory):
    """Analyze the given file and calculate the feature selection experiments for standard random forest.

    Args:
        file (Path): The file to analyze.
        result_base_path (Path): The path of the directory to store the results.
        path_to_random_noise_directory: The path of the directory with the original random noise data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the experiment id is invalid.
    """
    # extract the experiment id from the file name
    experiment_id = file.stem.split("_result_dict")[0]
    print("id", experiment_id)
    # check if the experiment id is a string
    assert isinstance(experiment_id, str), "Experiment id is not a string."
    # extract the data name from the experiment id
    if "colon" in experiment_id:
        data_name = "colon"
    elif "prostate" in experiment_id:
        data_name = "prostate"
    elif "leukemia" in experiment_id:
        data_name = "leukemia_big"
    elif "random" in experiment_id:
        data_name = "random_noise"
    else:
        raise ValueError(f"Invalid experiment id {experiment_id}.")
    print(data_name)
    # load previous result for the given experiment
    result_dict_path = Path(f"{result_base_path}/{experiment_id}_result_dict.pkl")
    if result_dict_path.exists():
        with open(result_dict_path, "rb") as result_file:
            result_dict = pickle.load(result_file)
    else:
        # throw exception if the file does not exist
        raise FileNotFoundError(f"File {result_dict_path} not found.")
    # define meta data for the experiment
    assert "reverse_random_forest_meta_data" in result_dict, "Missing reverse random forest meta data."
    meta_data_dict = {
        "git_commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "experiment_id": experiment_id,
        "data_name": data_name,
        # seed to shuffle the indices of the samples of the data set:
        "shuffle_seed": result_dict["reverse_random_forest_meta_data"]["shuffle_seed"],
        "n_cpus": result_dict["reverse_random_forest_meta_data"]["n_cpus"],
        # random seed for reproducibility of random forest:
        "random_state": result_dict["reverse_random_forest_meta_data"]["random_seeds"][0],
        "verbose_optuna": True,
        "n_trials_optuna": 80,
        "max_trees_random_forest": 2000,
    }
    if "random_noise" in file.name:
        # meta_data_dict["data_shape_random_noise"] = (62, 2000)
        # # The path to the directory where generated random noise is stored.
        # meta_data_dict["path_for_random_noise"] = f"{result_base_path}/random_noise"
        meta_data_dict["data_shape_random_noise"] = result_dict["reverse_random_forest_meta_data"][
            "data_shape_random_noise"
        ]
        assert path_to_random_noise_directory is not None
        if "lognormal" in file.name:
            meta_data_dict["path_for_random_noise"] = (
                f"{path_to_random_noise_directory}/random_noise_lognormal_30_2000.csv"
            )
        elif "lognormal" not in file.name:
            meta_data_dict["path_for_random_noise"] = (
                f"{path_to_random_noise_directory}/random_noise_normal_30_2000.csv"
            )
    # load data for the experiment with balanced train sample size
    data_df, _ = load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    print("number of samples", data_df.shape[0], "number of features", data_df.shape[1] - 1)
    # drop old data for standard random forest
    if "standard_random_forest" in result_dict:
        result_dict.pop("standard_random_forest")
        result_dict.pop("standard_random_forest_meta_data")
    # calculate raw feature subset data for standard random forest
    result_dict["standard_random_forest"] = cross_validation.cross_validate(
        data_df, meta_data_dict, calculate_feature_importance
    )
    result_dict["standard_random_forest_meta_data"] = meta_data_dict
    # save results
    result_dict_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_rf_result_dict.pkl")
    with open(result_dict_path, "wb") as result_file:
        pickle.dump(result_dict, result_file, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """Main function for calculating baseline feature selection experiments for standard random forest."""
    # parse result path from input
    result_base_path = Path(sys.argv[1])
    print("result data_path: ", result_base_path)

    # check if second argument was given
    path_to_random_noise_directory = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"Path to random noise data directory: {path_to_random_noise_directory}")

    # iterate over all files in the directory
    for file in result_base_path.iterdir():
        # # exclude files that are not related to the random noise data set
        # # or the ranger random forest
        # if "random" in file.name or "rf" not in file.name:
        #     continue
        # if not ("random" in file.name and "shuffle_seed_None" in file.name):
        #     print("continue: ", file.name)
        #     continue
        # print("file", file.name)

        analyze_file(file, result_base_path, path_to_random_noise_directory)


if __name__ == "__main__":
    main()
