import os

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.data_loader_tools import (
    balance_sample_size_of_hold_out_data,
    balance_sample_size_of_hold_out_data_single_df,
    convert_to_single_df,
    generate_random_noise_data,
    get_indices_for_selected_and_deselected_samples,
    shuffle,
)

# Create a random number generator
rng = np.random.default_rng()


# cleanup the generated files after testing
@pytest.fixture(autouse=True)
def cleanup_generated_files():
    # Code to run before each test
    yield
    # Code to run after each test
    for file in os.listdir("."):
        if file.startswith(("empty", "single_class")):
            os.remove(file)


def test_balance_sample_size_of_hold_out_data_single_df_empty():
    hold_out_data_df = pd.DataFrame()
    with pytest.raises(AssertionError) as e:
        balance_sample_size_of_hold_out_data_single_df(hold_out_data_df, shuffle_seed=42, number_of_balanced_samples=14)
    assert str(e.value) == "Number of balanced samples 14 is smaller than the number of samples in the hold out data 0."


def test_balance_sample_size_of_hold_out_data_single_df():
    data = pd.DataFrame(rng.standard_normal((20, 2000)))  # 20 samples, 2000 features
    label = pd.Series([0] * 10 + [1] * 10)
    hold_out_data_df = convert_to_single_df(data, label)
    balanced_df = balance_sample_size_of_hold_out_data_single_df(
        hold_out_data_df, shuffle_seed=42, number_of_balanced_samples=14
    )
    assert balanced_df["label"].value_counts().tolist() == [7, 7]


def test_balance_sample_size_of_hold_out_data():
    data = pd.DataFrame(rng.standard_normal((20, 2000)))
    label = pd.Series([0] * 10 + [1] * 10)
    hold_out_data_df = convert_to_single_df(data, label)
    balanced_df, balanced_label = balance_sample_size_of_hold_out_data(
        hold_out_data_df, shuffle_seed=42, number_of_balanced_samples=14
    )
    assert balanced_label.value_counts().tolist() == [7, 7]


def test_convert_to_single_df():
    x = pd.DataFrame(rng.standard_normal((30, 2000)))
    y = pd.Series([0] * 15 + [1] * 15)
    test_df = convert_to_single_df(x, y)
    assert test_df.shape == (30, 2001)
    assert test_df.columns[0] == "label"
    assert all(test_df["label"].isin([0, 1]))


def test_get_indices_for_selected_and_deselected_samples():
    label = pd.Series([0] * 20 + [1] * 20)
    train_indices, test_indices = get_indices_for_selected_and_deselected_samples(label, 0, 15)
    assert len(train_indices) == 15
    assert len(test_indices) == 5


def test_shuffle():
    data = pd.DataFrame(rng.standard_normal((40, 2000)))
    label = pd.Series([0] * 20 + [1] * 20)
    shuffled_data, shuffled_label = shuffle(data, label, shuffle_seed=42)
    assert shuffled_data.shape == data.shape
    assert shuffled_label.shape == label.shape
    assert not shuffled_data.equals(data)


def test_generate_random_noise_data_with_normal_distribution():
    data_shape = (30, 2000)
    data_df, _ = generate_random_noise_data(data_shape, distribution="normal")
    assert data_df.shape == (30, 2001)
    assert data_df["label"].value_counts().tolist() == [15, 15]


def test_generate_random_noise_data_with_lognormal_distribution():
    data_shape = (30, 2000)
    data_df, _ = generate_random_noise_data(data_shape, distribution="lognormal")
    assert data_df.shape == (30, 2001)
    assert data_df["label"].value_counts().tolist() == [15, 15]


def test_load_train_holdout_data_for_balanced_train_sample_size_random_noise():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "data_name": "random_noise_lognormal",
        "shuffle_seed": 42,
        "path_for_random_noise": "../random_noise_data/random_noise_lognormal_(30, 2000).csv",
    }
    train_data_df, hold_out_data_df = load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert train_data_df.shape[0] == 30
    assert hold_out_data_df.empty
    assert train_data_df["label"].value_counts().tolist() == [15, 15]


def test_balance_sample_size_of_hold_out_data_single_df_uneven_samples():
    data = pd.DataFrame(rng.standard_normal((21, 2000)))  # 21 samples, 2000 features
    label = pd.Series([0] * 10 + [1] * 11)
    hold_out_data_df = convert_to_single_df(data, label)
    balanced_df = balance_sample_size_of_hold_out_data_single_df(
        hold_out_data_df, shuffle_seed=42, number_of_balanced_samples=14
    )
    assert balanced_df["label"].value_counts().tolist() == [7, 7]


def test_convert_to_single_df_empty():
    x = pd.DataFrame()
    y = pd.Series(dtype=int)
    test_df = convert_to_single_df(x, y)
    assert test_df.empty


def test_shuffle_fixed_seed():
    data = pd.DataFrame(rng.standard_normal((40, 2000)))
    label = pd.Series([0] * 20 + [1] * 20)
    shuffled_data_1, shuffled_label_1 = shuffle(data, label, shuffle_seed=42)
    shuffled_data_2, shuffled_label_2 = shuffle(data, label, shuffle_seed=42)
    pd.testing.assert_frame_equal(shuffled_data_1, shuffled_data_2)
    pd.testing.assert_series_equal(shuffled_label_1, shuffled_label_2)


def test_load_train_holdout_data_for_balanced_train_sample_size_colon():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    train_data_df, hold_out_data_df = load_train_holdout_data_for_balanced_train_sample_size(
        {
            "data_name": "colon",
            "shuffle_seed": None,
        }
    )
    assert hold_out_data_df.columns[0] == "label"
    assert train_data_df.columns[0] == "label"
    assert train_data_df["label"].value_counts().tolist() == [
        15,
        15,
    ], f"{train_data_df['label'].value_counts().tolist()}"
    assert train_data_df.shape[1] == hold_out_data_df.shape[1]  # label column
    assert train_data_df.shape[0] + hold_out_data_df.shape[0] == 62


def test_load_train_holdout_data_for_balanced_train_sample_size_prostate():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "data_name": "prostate",
        "shuffle_seed": None,
    }
    train_data_df, hold_out_data_df = load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert hold_out_data_df.columns[0] == "label"
    assert train_data_df.columns[0] == "label"
    assert train_data_df["label"].value_counts().tolist() == [15, 15]

    assert train_data_df.shape[1] == hold_out_data_df.shape[1]  # label column
    assert train_data_df.shape[0] + hold_out_data_df.shape[0] == 102


def test_load_train_holdout_data_for_balanced_train_sample_size_leukemia_big():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "data_name": "leukemia_big",
        "shuffle_seed": None,
    }
    train_data_df, hold_out_data_df = load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert hold_out_data_df.columns[0] == "label"
    assert train_data_df.columns[0] == "label"
    assert train_data_df["label"].value_counts().tolist() == [15, 15]

    assert train_data_df.shape[1] == hold_out_data_df.shape[1]  # label column
    assert (
        train_data_df.shape[0] + hold_out_data_df.shape[0] == 72
    ), f"{train_data_df.shape[0] + hold_out_data_df.shape[0]}"


def test_load_train_holdout_data_for_balanced_train_sample_size_invalid_data_name():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "data_name": "invalid_data_name",
        "shuffle_seed": 42,
    }
    with pytest.raises(ValueError) as e:
        load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert str(e.value) == "Data name invalid_data_name not found or path to dataset is missing in meta_data_dict."


def test_load_train_holdout_data_for_balanced_train_sample_size_missing_data_name():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "shuffle_seed": 42,
    }
    with pytest.raises(AssertionError) as e:
        load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert str(e.value) == "Data name is missing in meta_data_dict."


def test_load_train_holdout_data_for_balanced_train_sample_size_single_class():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "data_name": "single_class_test",
        "shuffle_seed": None,
    }
    # Mocking a dataset with a single class
    data = pd.DataFrame({"label": [0] * 30})
    # insert random data
    data = pd.concat([data, pd.DataFrame(rng.standard_normal((30, 2000)))], axis=1)
    data.to_csv("single_class_test.csv", index=False)
    meta_data_dict["path_to_local_dataset"] = "single_class_test.csv"
    with pytest.raises(AssertionError) as e:
        load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert str(e.value) == "Class label 1 not found in label series."


def test_load_train_holdout_data_for_balanced_train_sample_size_no_shuffle_seed():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "data_name": "colon",
    }
    with pytest.raises(AssertionError) as e:
        load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert str(e.value) == "Shuffle seed is missing in meta_data_dict."


def test_load_train_holdout_data_for_balanced_train_sample_size_invalid_shuffle_seed():
    from feature_selection_benchmark.data_loader_tools import load_train_holdout_data_for_balanced_train_sample_size

    meta_data_dict = {
        "data_name": "colon",
        "shuffle_seed": "invalid_seed",
    }
    with pytest.raises(ValueError) as e:
        load_train_holdout_data_for_balanced_train_sample_size(meta_data_dict)
    assert str(e.value) == (
        "random_state must be an integer, array-like, a BitGenerator, Generator, a numpy RandomState, or None"
    )
