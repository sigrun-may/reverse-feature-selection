# -*- coding: utf-8 -*-
# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data loader tools."""
from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd
from mltb2.data import load_colon, load_leukemia_big, load_prostate


def load_data_with_standardized_sample_size(
    data: Literal["colon", "prostate", "leukemia_big"] = "colon"
) -> pd.DataFrame:
    """Load data with standardized sample size.

    The data is loaded and parsed from the internet (see mltb2.data). The function standardizes the sample size to
    the first 15 samples of each class.

    Args:
        data: Name of the data to load. Possible options are "colon", "prostate", "leukemia_big".

    Returns:
        Dataframe containing labels and data.
    """
    # generate function from string
    load_data_function = globals()[f"load_{data}"]

    # standardize sample size to balanced data
    label, data = load_data_function()
    x, y = standardize_sample_size(data, label)
    return convert_to_single_df(x, y)


def convert_to_single_df(x, y):
    """Merge data and labels into a single dataframe and set column names.

    Args:
        x: Data.
        y: Labels.

    Returns:
        Dataframe containing labels in the first column and data.
    """
    data_df = pd.DataFrame(x)
    # convert integer column names to string
    data_df.columns = [f"f_{i}" for i in range(data_df.shape[1])]
    data_df.insert(loc=0, column="label", value=y)

    # reset index for cross validation splits
    data_df = data_df.reset_index(drop=True)
    assert data_df.columns[0] == "label"
    return data_df


def load_train_test_data_for_standardized_sample_size(
    data_name: Literal["colon", "prostate", "leukemia_big"] = "colon"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load remaining test data from standardized train sample size.

    The data is loaded and parsed from the internet (see mltb2.data). The function selects all samples but the first 15
     of each class as test data.

    Args:
        data_name: Name of the data to load. Possible options are "colon", "prostate", "leukemia_big".

    Returns:
        Dataframe containing labels and data.
    """
    # generate function from string
    load_data_function = globals()[f"load_{data_name}"]

    # standardize sample size to balanced data
    label, data = load_data_function()

    # find remaining samples for each class
    indices: List[int] = []
    counter_0 = 0
    counter_1 = 0
    for i in range(label.shape[0]):
        # select samples with label 0
        if label.iloc[i] == 0:
            counter_0 += 1
            if counter_0 > 15:
                indices.append(i)

        # select samples with label 1
        if label.iloc[i] == 1:
            counter_1 += 1
            if counter_1 > 15:
                indices.append(i)

    assert len(indices) == data.shape[0] - 30, f"{len(indices)} != {data.shape[0]} - 30"
    test_data_df = data.iloc[indices]
    y_test = label.iloc[indices]
    assert test_data_df.shape[0] == data.shape[0] - 30, f"{test_data_df.shape[0]} != {data.shape[0] - 30}"
    assert len(y_test) == test_data_df.shape[0]
    assert test_data_df.shape[1] == data.shape[1]

    test_data_df = convert_to_single_df(test_data_df, y_test)
    x, y = standardize_sample_size(data, label)
    train_data_df = convert_to_single_df(x, y)

    assert train_data_df.shape[1] == test_data_df.shape[1]
    assert train_data_df.shape[0] + test_data_df.shape[0] == data.shape[0]

    return test_data_df, train_data_df



def standardize_sample_size(data, label) -> Tuple[pd.DataFrame, pd.Series]:
    """Reduce samples to 15 for each class.

    Returns:
        Tuple containing balanced data and corresponding labels.
    """
    # reduce samples to 15 for each class
    indices: List[int] = []
    for i in range(label.shape[0]):
        if label.iloc[i] == 0:
            indices.append(i)
        if len(indices) == 15:
            break

    for i in range(label.shape[0]):
        if label.iloc[i] == 1:
            indices.append(i)
        if len(indices) == 30:
            break

    data = data.iloc[indices]
    label = label.iloc[indices]
    assert data.shape[0] == 30
    assert len(label) == 30
    return data, label


def get_data_df(meta_data_dict: dict, generate_random_noise: bool = False, path_for_random_noise: str = "") -> pd.DataFrame:
    """Load data for the experiment.

    Args:
        meta_data_dict: The metadata related to the dataset and experiment.
        generate_random_noise: If True, generate random noise data for benchmarking.
        path_for_random_noise: The path to the directory where generated random noise is stored.

    Returns:
        The data as a pandas DataFrame with the label in the first column.
    """

    if "random" in meta_data_dict["data_name"]:
        # load artificial data
        data_df = pd.read_csv(f"../data/{meta_data_dict['experiment_id']}_data_df.csv")
    elif generate_random_noise:
        # generate random noise data for benchmarking
        import numpy as np
        rnd = np.random.default_rng(seed=42)
        data_df = pd.DataFrame(rnd.normal(scale=2, size=(30, 2000)), columns=[f"f_{i}" for i in range(2000)])
        # generate array as binary balanced target with two classes for 30 samples
        label_array = np.concatenate([np.zeros(15), np.ones(15)])
        data_df.insert(0, "label", label_array)
        # save data_df as csv
        assert path_for_random_noise != "", "Result base path is empty."
        data_df_path = Path(f"{path_for_random_noise}/{meta_data_dict['experiment_id']}_data_df.csv")
        data_df.to_csv(data_df_path, index=False)
    else:
        # load data
        assert meta_data_dict["data_name"] in ["colon", "prostate", "leukemia_big"], "Invalid data name."
        data_df = load_data_with_standardized_sample_size(meta_data_dict["data_name"])
    return data_df
    # return pd.read_csv(f"../../data/artificial_biomarker_data_2.csv").iloc[:, list(range(10)) + list(range(-7, 0))]