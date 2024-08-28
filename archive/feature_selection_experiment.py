# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Main script for cacluating the raw results of the feature selection."""

import cross_validation
import git

from data_loader_tools import load_data_with_standardized_sample_size


def load_data(meta_data_dict):
    """Load the data based on the meta data dictionary."""

    if "random" in meta_data_dict["data_name"]:
        # load artificial data
        import pandas as pd

        data_df = pd.read_csv(f"../data/{meta_data_dict['experiment_id']}_data_df.csv")

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
    else:
        # load data
        assert meta_data_dict["data_name"] in ["colon", "prostate", "leukemia_big"], "Invalid data name."
        data_df = load_data_with_standardized_sample_size(meta_data_dict["data_name"])
    print("number of samples", data_df.shape[0], "number of features", data_df.shape[1] - 1)

    return data_df


def run_experiment(meta_data_dict, result_dict):
    """Calculate the raw results of the feature selection."""

    # save git commit hash
    git_repository = git.Repo(search_parent_directories=True)
    meta_data_dict["git_commit_hash"] = git_repository.head.object.hexsha

    if "random" in meta_data_dict["data_name"]:
        # load artificial data
        import pandas as pd

        data_df = pd.read_csv(f"../data/{meta_data_dict['experiment_id']}_data_df.csv")

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
    else:
        # load data
        assert meta_data_dict["data_name"] in ["colon", "prostate", "leukemia_big"], "Invalid data name."
        data_df = load_data_with_standardized_sample_size(meta_data_dict["data_name"])
    print("number of samples", data_df.shape[0], "number of features", data_df.shape[1] - 1)

    if "random_seeds" in meta_data_dict:
        # calculate raw feature subset data for reverse random forest
        assert "reverse_random_forest" not in result_dict, "Reverse random forest results are already present."
        from reverse_feature_selection.reverse_rf_random import select_feature_subset

        result_dict["reverse_random_forest"] = cross_validation.cross_validate(
            data_df, meta_data_dict, select_feature_subset
        )
        result_dict["reverse_random_forest_meta_data"] = meta_data_dict
    else:
        # calculate raw feature subset data for standard random forest
        # del result_dict["standard_random_forest"]
        assert "standard_random_forest" not in result_dict, "Standard random forest results are already present."
        from standard_rf import calculate_feature_importance

        result_dict["standard_random_forest"] = cross_validation.cross_validate(
            data_df, meta_data_dict, calculate_feature_importance
        )
        result_dict["standard_random_forest_meta_data"] = meta_data_dict
