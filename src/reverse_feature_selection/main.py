import datetime
import pickle
import sys
from pathlib import Path

import toml
import git

import cross_validation


def main():
    # parse result path from input
    result_base_path = Path(sys.argv[1])
    print("data_path: ", result_base_path)

    # parse settings from toml
    with open("./settings.toml", "r") as file:
        meta_data_dict = toml.load(file)

    # save git commit hash
    git_repository = git.Repo(search_parent_directories=True)
    commit_sha = git_repository.head.object.hexsha
    meta_data_dict["git_commit_hash"] = commit_sha

    # print number of available CPUs
    import multiprocessing

    print("Number of CPUs available: ", multiprocessing.cpu_count())

    # # load previous cross-validation indices for the given experiment, if available
    result_dict_path = Path(f"{result_base_path}/{meta_data_dict['experiment_id']}_result_dict.pkl")
    # if result_dict_path.exists():
    #     with open(result_dict_path, "rb") as file:
    #         result_dict = pickle.load(file)
    # else:
    #     result_dict = {}
    result_dict = {}
    # # load artificial data
    # import pandas as pd
    # data_df = pd.read_csv(meta_data_dict["data"]["path"]).iloc[:, :221]
    # # shorten artificial data for faster testing
    # data_01 = data_df.iloc[:, 0:30]
    # data_02 = data_df.iloc[:, 200:220]
    # data_df = pd.concat([data_01, data_02], join="outer", axis=1)

    # data loaders
    from data_loader_tools import load_data_with_standardized_sample_size

    data_df = load_data_with_standardized_sample_size("colon")

    assert data_df.columns[0] == "label"
    print(data_df.shape)
    print("number of samples", data_df.shape[0], "number of features", data_df.shape[1] - 1)

    # calculate raw feature subset data
    start_time = datetime.datetime.utcnow()
    result_dict = cross_validation.cross_validate(data_df, meta_data_dict, result_dict)
    end_time = datetime.datetime.utcnow()
    print(end_time - start_time)

    # save results
    result_base_path.mkdir(parents=True, exist_ok=True)
    with open(result_dict_path, "wb") as file:
        pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
