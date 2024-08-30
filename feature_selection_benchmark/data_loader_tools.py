# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data loader tools."""
from pathlib import Path

import pandas as pd
from mltb2.data import load_colon, load_leukemia_big, load_prostate


def standardize_sample_size_of_hold_out_data(hold_out_data_df: pd.DataFrame, shuffle_seed: int) -> pd.DataFrame:
    """Standardize the sample size of the hold out data, to ensure balanced classes.

    Args:
        hold_out_data_df: The hold out data as a pandas DataFrame with the label in the first column.
        shuffle_seed: Seed for shuffling.

    Returns:
        The standardized data as a pandas DataFrame with balanced classes: 7 samples for each class.
    """
    # shuffle data before selecting samples, if shuffle_seed is set
    unlabeled_hold_out_data_df, label_series = shuffle(
        hold_out_data_df.iloc[:, 1:], hold_out_data_df["label"], shuffle_seed
    )
    max_samples = 7
    hold_out_test_indices_0, _ = get_indices_for_class(label_series, 0, max_samples)
    hold_out_test_indices_1, _ = get_indices_for_class(label_series, 1, max_samples)

    hold_out_test_indices = hold_out_test_indices_0 + hold_out_test_indices_1
    balanced_hold_out_test_data_df = convert_to_single_df(
        unlabeled_hold_out_data_df.iloc[hold_out_test_indices], label_series.iloc[hold_out_test_indices]
    )
    return balanced_hold_out_test_data_df


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


def get_indices_for_class(label_series: pd.Series, class_label: int, max_samples: int) -> tuple[list[int], list[int]]:
    """Get train and test indices for a specific class label.

    Args:
        label_series: Series containing labels.
        class_label: The class label to filter.
        max_samples: Maximum number of samples to include in the train set.

    Returns:
        Tuple containing train indices and test indices.
    """
    # check if class label is in label series
    assert class_label in label_series.unique(), f"Class label {class_label} not found in label series."

    # check if max_samples is smaller than the number of samples for the class label
    assert (
        max_samples < label_series[label_series == class_label].shape[0]
    ), f"Number of samples for class label {class_label} is smaller than max_samples."

    train_indices = []
    test_indices = []
    counter = 0

    for i in range(label_series.shape[0]):
        if label_series.iloc[i] == class_label:
            if counter >= max_samples:
                test_indices.append(i)
            else:
                train_indices.append(i)
            counter += 1
    assert len(train_indices) == max_samples, f"{len(train_indices)} != {max_samples}"
    assert len(test_indices) == label_series[label_series == class_label].shape[0] - max_samples
    return train_indices, test_indices


def shuffle(data: pd.DataFrame, label: pd.Series, shuffle_seed: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Shuffle data and labels.

    Args:
        data: Data.
        label: Labels.
        shuffle_seed: Seed for shuffling.

    Returns:
        Tuple containing shuffled data and corresponding labels
    """
    if shuffle_seed is not None:
        data_shape = data.shape
        data_index = data.index

        # shuffle data before selecting samples
        data = data.sample(frac=1, random_state=shuffle_seed)
        label = label.loc[data.index]

        assert data.index.equals(label.index)
        assert data.shape[0] == label.shape[0]
        assert data_shape == data.shape
        assert not data_index.equals(data.index)
    return data, label


def load_train_test_data_for_standardized_sample_size(meta_data_dict: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load remaining test data from standardized train sample size.

    The data is loaded and parsed from the internet (see mltb2.data). The function selects all samples but the first 15
     of each class as test data. If the shuffle_seed is set, the data is shuffled before selecting the samples.

    Args:
        meta_data_dict: The metadata related to the dataset and experiment.
            Must contain the key "data_name" and "shuffle_seed". Data name must be "colon", "prostate" or
            "leukemia_big". "shuffle_seed" is an integer or None.

    Returns:
        Tuple containing train data and hold out set as test data.
    """
    # generate function from string
    load_data_function = globals()[f"load_{meta_data_dict['data_name']}"]

    # standardize sample size to balanced data
    label, data = load_data_function()

    # shuffle data before selecting samples, if shuffle_seed is set
    data, label = shuffle(data, label, meta_data_dict["shuffle_seed"])

    train_indices_0, test_indices_0 = get_indices_for_class(label, 0, 15)
    train_indices_1, test_indices_1 = get_indices_for_class(label, 1, 15)

    train_indices = train_indices_0 + train_indices_1
    test_indices = test_indices_0 + test_indices_1

    assert len(test_indices) == data.shape[0] - 30, f"{len(test_indices)} != {data.shape[0]} - 30"
    test_data_df = data.iloc[test_indices]
    y_test = label.iloc[test_indices]
    assert test_data_df.shape[0] == data.shape[0] - 30, f"{test_data_df.shape[0]} != {data.shape[0] - 30}"
    assert len(y_test) == test_data_df.shape[0]
    assert test_data_df.shape[1] == data.shape[1]

    assert len(train_indices) == 30
    train_data_df = data.iloc[train_indices]
    y_train = label.iloc[train_indices]
    assert train_data_df.shape[0] == 30
    assert len(y_train) == 30
    assert train_data_df.shape[1] == data.shape[1]

    test_data_df = convert_to_single_df(test_data_df, y_test)
    train_data_df = convert_to_single_df(train_data_df, y_train)
    assert test_data_df.columns[0] == "label"
    assert train_data_df.columns[0] == "label"
    assert train_data_df["label"].value_counts().tolist() == [
        15,
        15,
    ], f"{train_data_df['label'].value_counts().tolist()}"
    assert train_data_df.shape[1] == test_data_df.shape[1] == data.shape[1] + 1  # label column
    assert train_data_df.shape[0] + test_data_df.shape[0] == data.shape[0]

    return train_data_df, test_data_df


def load_data_df(
    meta_data_dict: dict, generate_random_noise: bool = False, path_for_random_noise: str = ""
) -> pd.DataFrame:
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
        data_df, _ = load_train_test_data_for_standardized_sample_size(meta_data_dict)
        assert data_df.shape[0] == 30, f"Number of samples is not 30: {data_df.shape[0]}"
    return data_df
    # return pd.read_csv(f"../../data/artificial_biomarker_data_2.csv").iloc[:, list(range(10)) + list(range(-7, 0))]
