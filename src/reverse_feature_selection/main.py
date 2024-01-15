import datetime
import os
import pickle
import sys
from pathlib import Path

import toml
import git

from cross_validation import CrossValidator
from data_loader_tools import load_data_with_standardized_sample_size


def main():
    # parse settings from toml
    with open("./settings.toml", "r") as file:
        meta_data = toml.load(file)

    # save git commit hash
    git_repository = git.Repo(search_parent_directories=True)
    commit_sha = git_repository.head.object.hexsha
    meta_data["git_commit_hash"] = commit_sha

    # # load artificial data
    # import pandas as pd
    # data_df = pd.read_csv(meta_data["data"]["path"]).iloc[:, :50]
    # assert data_df.columns[0] == "label"

    # # shorten artificial data for faster testing
    # data_01 = data_df.iloc[:, 0:10]
    # data_02 = data_df.iloc[:, 200:800]
    # data_df = pd.concat([data_01, data_02], join="outer", axis=1)

    # data loaders
    data_df = load_data_with_standardized_sample_size("colon")
    print(data_df.shape)

    start_time = datetime.datetime.utcnow()

    cross_validator = CrossValidator(data_df, meta_data)
    result_dict = cross_validator.cross_validate()

    end_time = datetime.datetime.utcnow()
    print(end_time - start_time)

    # save results
    # check if program is running on cluster
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = Path(f"./")
    print("data_path: ", data_path)

    # Create path for saving the results
    result_base_path = Path(os.path.join(data_path, "results"))

    # Create directory for saving the results
    result_base_path.mkdir(parents=True, exist_ok=True)
    with open(f"{result_base_path}/{meta_data['experiment_id']}_result_dict.pkl", "wb") as file:
        pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # with open("/home/sigrun/PycharmProjects/reverse_feature_selection/results/delete_me_result_dict.pkl", "rb") as file:
    #     test = pickle.load(file)
    # print(test)
    main()