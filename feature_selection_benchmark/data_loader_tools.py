# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data loader tools."""
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# load modules dynamically with globals()
from mltb2.data import load_colon, load_leukemia_big, load_prostate  # noqa: F401


def balance_sample_size_of_hold_out_data_single_df(
    hold_out_data_df: pd.DataFrame, shuffle_seed: int | None, number_of_balanced_samples: int
) -> pd.DataFrame:
    """Standardize the sample size of the hold out data, to ensure balanced classes.

    Args:
        hold_out_data_df: The hold out data as a pandas DataFrame with the label in the first column.
        shuffle_seed: Seed for shuffling.
        number_of_balanced_samples: Number of balanced samples for both classes. Must be an even number.

    Returns:
        The standardized data as a pandas DataFrame with balanced classes: 7 samples for each class.
    """
    # check if number of samples is an even number
    assert (
        number_of_balanced_samples % 2 == 0
    ), f"Number of balanced samples is not an even number: {number_of_balanced_samples}"

    # check if number_of_balanced_samples is smaller than the number of samples in the hold out data
    assert number_of_balanced_samples <= hold_out_data_df.shape[0], (
        f"Number of balanced samples {number_of_balanced_samples} is smaller than the number of samples "
        f"in the hold out data {hold_out_data_df.shape[0]}."
    )

    # check if number_of_balanced_samples/2 is smaller than the number of samples for each class
    assert number_of_balanced_samples / 2 <= hold_out_data_df["label"].value_counts().min(), (
        f"Number of balanced samples {number_of_balanced_samples / 2} for each class is smaller than the number of "
        f"samples for each class in the hold out data {hold_out_data_df['label'].value_counts().min()}."
    )

    balanced_hold_out_test_data_df, balanced_hold_out_test_label_series = balance_sample_size_of_hold_out_data(
        hold_out_data_df, shuffle_seed
    )
    assert len(balanced_hold_out_test_label_series) == balanced_hold_out_test_data_df.shape[0]
    assert len(balanced_hold_out_test_label_series) == number_of_balanced_samples
    # check if the class labels are balanced
    assert balanced_hold_out_test_label_series.value_counts().tolist() == [
        number_of_balanced_samples / 2,
        number_of_balanced_samples / 2,
    ]

    balanced_hold_out_test_data_df = convert_to_single_df(
        balanced_hold_out_test_data_df, balanced_hold_out_test_label_series
    )
    return balanced_hold_out_test_data_df


def balance_sample_size_of_hold_out_data(
    hold_out_data_df: pd.DataFrame, shuffle_seed: int | None, number_of_balanced_samples: int = 14
) -> tuple[pd.DataFrame, pd.Series]:
    """Standardize the sample size of the hold out data, to ensure balanced classes.

    Args:
        hold_out_data_df: The hold out data as a pandas DataFrame with the label in the first column.
        shuffle_seed: Seed for shuffling.
        number_of_balanced_samples: Number of balanced samples for both classes. Must be an even number.

    Returns:
        The standardized data as a pandas DataFrame with balanced classes: 7 samples for each class.
    """
    # check if number of samples is an even number
    assert (
        number_of_balanced_samples % 2 == 0
    ), f"Number of balanced samples is not an even number: {number_of_balanced_samples}"

    # shuffle data before selecting samples
    unlabeled_hold_out_data_df, label_series = shuffle(
        hold_out_data_df.iloc[:, 1:], hold_out_data_df["label"], shuffle_seed
    )
    number_of_samples = int(number_of_balanced_samples / 2)
    hold_out_test_indices_0, _ = get_indices_for_selected_and_deselected_samples(label_series, 0, number_of_samples)
    hold_out_test_indices_1, _ = get_indices_for_selected_and_deselected_samples(label_series, 1, number_of_samples)

    hold_out_test_indices = hold_out_test_indices_0 + hold_out_test_indices_1
    assert len(hold_out_test_indices) == number_of_balanced_samples

    return unlabeled_hold_out_data_df.iloc[hold_out_test_indices], label_series.iloc[hold_out_test_indices]


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


def get_indices_for_selected_and_deselected_samples(
    label_series: pd.Series, class_label: int, number_of_samples: int
) -> tuple[list[int], list[int]]:
    """Get selected and deselected indices for a specific class label.

    Args:
        label_series: Series containing labels.
        class_label: The class label to filter.
        number_of_samples: Number of samples from the given class_label to select.

    Returns:
        Tuple containing selected and deselected indices.
    """
    # check if class label is in label series
    assert class_label in label_series.unique(), f"Class label {class_label} not found in label series."

    # check if number_of_samples is smaller than the number of samples for the class label
    assert number_of_samples <= label_series[label_series == class_label].shape[0], (
        f"Number of samples {label_series[label_series == class_label].shape[0]} for class label {class_label} "
        f"is smaller than number_of_samples {number_of_samples}."
    )

    selected_indices = []
    deselected_indices = []
    counter = 0

    for i in range(label_series.shape[0]):
        if label_series.iloc[i] == class_label:
            if counter >= number_of_samples:
                deselected_indices.append(i)
            else:
                selected_indices.append(i)
            counter += 1
    assert len(selected_indices) == number_of_samples, f"{len(selected_indices)} != {number_of_samples}"
    assert len(deselected_indices) == label_series[label_series == class_label].shape[0] - number_of_samples
    return selected_indices, deselected_indices


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
        label_index = label.index

        # shuffle data before selecting samples
        shuffled_data = data.sample(frac=1, random_state=shuffle_seed)
        shuffled_label = label.loc[shuffled_data.index]

        assert shuffled_data.index.equals(shuffled_label.index)
        assert shuffled_data.shape[0] == shuffled_label.shape[0]
        assert data_shape == shuffled_data.shape
        assert not data_index.equals(shuffled_data.index)
        assert not label_index.equals(shuffled_label.index)
        return shuffled_data, shuffled_label
    return data, label


def load_train_holdout_data_for_balanced_train_sample_size(
    meta_data_dict: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load remaining test data from standardized train sample size.

    The data is loaded and parsed from the internet (see mltb2.data). The function selects all samples but the first 15
     of each class as test data. If the shuffle_seed is set, the data is shuffled before selecting the samples.

    Args:
        meta_data_dict: The metadata related to the dataset and experiment.
            Must contain the key "data_name", "shuffle_seed" or "path_for_random_noise" and in case of local data
            to load the "path_to_local_dataset" .
            Data name must be "colon", "prostate" or "leukemia_big" for loading data from a defined online source.
            "shuffle_seed" must be an integer or None.

    Returns:
        Tuple containing train data and hold out set as test data.
    """
    assert "data_name" in meta_data_dict, "Data name is missing in meta_data_dict."
    assert "shuffle_seed" in meta_data_dict, "Shuffle seed is missing in meta_data_dict."

    # load data from mltb2.data
    if meta_data_dict["data_name"] in ["colon", "prostate", "leukemia_big"]:
        # generate function from string
        load_data_function = globals()[f"load_{meta_data_dict['data_name']}"]

        # load data
        label_series, data_df = load_data_function()

    # load generated random noise data
    elif "random" in meta_data_dict["data_name"]:
        assert "path_for_random_noise" in meta_data_dict
        # try to load random noise data if it exists
        data_df_path = Path(meta_data_dict["path_for_random_noise"])
        if data_df_path.exists():
            complete_data_df = pd.read_csv(data_df_path)
            label_series = complete_data_df["label"]
            data_df = complete_data_df.drop(columns=["label"])
        else:
            raise FileNotFoundError(f"Data file {data_df_path} not found.")

    # load local data
    elif "path_to_local_dataset" in meta_data_dict:
        # try to load the local data if it exists
        data_df_path = Path(meta_data_dict["path_to_local_dataset"])
        if data_df_path.exists():
            complete_data_df = pd.read_csv(data_df_path)
            label_series = complete_data_df["label"]
            data_df = complete_data_df.drop(columns=["label"])
        else:
            raise FileNotFoundError(f"Data file {data_df_path} not found.")
    else:
        raise ValueError(
            f"Data name {meta_data_dict['data_name']} not found or path to dataset is missing in meta_data_dict."
        )

    # check if data_df is not empty
    assert not data_df.empty, "Data frame is empty."

    # shuffle data before selecting samples, if shuffle_seed is set
    if meta_data_dict["shuffle_seed"] is not None:
        if "label" in data_df.columns:
            label_series = data_df["label"]
            data_df = data_df.drop(columns=["label"])
        data_df, label_series = shuffle(data_df, label_series, meta_data_dict["shuffle_seed"])

    train_indices_0, hold_out_data_indices_0 = get_indices_for_selected_and_deselected_samples(label_series, 0, 15)
    train_indices_1, hold_out_data_indices_1 = get_indices_for_selected_and_deselected_samples(label_series, 1, 15)

    train_indices = train_indices_0 + train_indices_1
    hold_out_data_indices = hold_out_data_indices_0 + hold_out_data_indices_1

    assert (
        len(hold_out_data_indices) == data_df.shape[0] - 30
    ), f"{len(hold_out_data_indices)} != {data_df.shape[0]} - 30"
    hold_out_data_df = data_df.iloc[hold_out_data_indices]
    y_test = label_series.iloc[hold_out_data_indices]
    assert hold_out_data_df.shape[0] == data_df.shape[0] - 30, f"{hold_out_data_df.shape[0]} != {data_df.shape[0] - 30}"
    assert len(y_test) == hold_out_data_df.shape[0]
    assert hold_out_data_df.shape[1] == data_df.shape[1]

    train_data_df = data_df.iloc[train_indices]
    y_train = label_series.iloc[train_indices]
    assert train_data_df.shape[0] == len(train_indices) == len(y_train) == 30
    assert train_data_df.shape[1] == data_df.shape[1]

    hold_out_data_df = convert_to_single_df(hold_out_data_df, y_test)
    train_data_df = convert_to_single_df(train_data_df, y_train)
    assert train_data_df.shape[1] == hold_out_data_df.shape[1] == data_df.shape[1] + 1  # label column
    assert train_data_df.shape[0] + hold_out_data_df.shape[0] == data_df.shape[0]

    return train_data_df, hold_out_data_df


def generate_random_noise_data(
    data_shape: tuple[int, int],
    distribution: Literal["normal", "lognormal"] = "normal",
    data_df_path: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Generate random noise data for benchmarking and save it as a CSV file.

    The data is standardized to a balanced sample size of 30 samples with the label in the first column. It is generated
        with a normal or lognormal distribution. If data_df_path is provided the generated data is saved as a
        CSV file with "random_noise_{distribution}_{data_shape}.csv" as filename.

    Args:
        data_shape: The shape of the generated data as a tuple of integers.
        distribution: The distribution of the random noise data. Must be "normal" or "lognormal".
        data_df_path: The path where the generated data will be saved if provided. If None, the data will not be saved.

    Returns:
        The generated data as a pandas DataFrame with the label in the first column.
    """
    rnd = np.random.default_rng(seed=42)
    data_name = f"random_noise_{distribution}_{data_shape}"
    if distribution == "lognormal":
        data = rnd.lognormal(mean=0, sigma=1, size=data_shape)
    elif distribution == "normal":
        data = rnd.normal(loc=0, scale=2, size=data_shape)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")
    data_df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(data_shape[1])])

    # generate array as binary balanced target with two classes
    number_of_elements_class_0 = int(data_shape[0] / 2)
    number_of_elements_class_1 = int(data_shape[0] - number_of_elements_class_0)
    label_array = np.concatenate([np.zeros(number_of_elements_class_0), np.ones(number_of_elements_class_1)])
    data_df.insert(0, "label", label_array)

    # save data_df as csv
    if data_df_path is not None:
        Path(data_df_path).mkdir(parents=True, exist_ok=True)
        data_df.to_csv(Path(f"{data_df_path}/{data_name}.csv"), index=False)

    # standardize sample size to balanced data of 30 samples
    class_indices_0, _ = get_indices_for_selected_and_deselected_samples(data_df["label"], 0, 15)
    class_indices_1, _ = get_indices_for_selected_and_deselected_samples(data_df["label"], 1, 15)
    data_df = convert_to_single_df(
        data_df.iloc[class_indices_0 + class_indices_1, 1:], data_df.iloc[class_indices_0 + class_indices_1, 0]
    )

    return data_df, data_name


# # generate_random_noise_data((30, 2000), "lognormal", "../random_noise_data")
# import matplotlib.pyplot as plt
#
# # Read the CSV file
# data_df = pd.read_csv("../random_noise_data/random_noise_normal_(30, 2000)")
#
# # Skip the first column (label)
# data = data_df.iloc[:, 1:]
#
# # Plot the histogram
# plt.hist(data.values.flatten(), bins=100, color="blue", edgecolor="black")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.title("Histogram of Random Noise Data")
#
# # Show the plot
# plt.show()
