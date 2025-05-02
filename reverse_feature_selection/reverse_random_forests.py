# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Reverse feature selection with random forest regressors."""

import logging
import math
import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_features_correlated_to_target_feature(
    train_df: pd.DataFrame, correlation_matrix_df: pd.DataFrame, target_feature: str, meta_data: dict
) -> pd.DataFrame:
    """Remove features from the training data that are correlated to the target feature.

    This function creates a mask for uncorrelated features based on the correlation threshold
    specified in the metadata. It then uses this mask to select the uncorrelated features from the training data.

    Args:
        train_df: The training data.
        correlation_matrix_df: The correlation matrix of the training data.
        target_feature: The name of the target feature.
        meta_data: The metadata related to the dataset and experiment. Must contain "train_correlation_threshold" to
            define the threshold for the absolute correlation to the target feature.

    Returns:
        The training data including the label with only the features uncorrelated to the target feature remaining.

    Raises:
        AssertionError: If no features uncorrelated to the target feature are found.
    """
    # Create a mask for uncorrelated features based on the correlation threshold
    uncorrelated_features_mask = (
        correlation_matrix_df[target_feature]
        .abs()
        .le(meta_data["train_correlation_threshold"], axis="index")
        # For a correlation matrix filled only with the lower half,
        # the first elements up to the diagonal would have to be read
        # with axis="index" and the further elements after the diagonal
        # with axis="column".
    )
    # Remove correlated features from the training data
    uncorrelated_train_df = train_df[train_df.columns[uncorrelated_features_mask]]

    assert len(uncorrelated_train_df.columns) > 1, "No features uncorrelated to the target feature found."

    # insert the 'label' as the first column if it is not already there
    if uncorrelated_train_df.columns[0] != "label":
        uncorrelated_train_df.insert(0, "label", train_df["label"])

    # Return the data frame with uncorrelated features
    return uncorrelated_train_df


def calculate_oob_errors(
    target_feature_name: str,
    train_df: pd.DataFrame,
    corr_matrix_df: pd.DataFrame,
    meta_data: dict,
) -> tuple[list | None, list | None, int]:
    """Calculate out-of-bag (OOB) error for training data first including the label and again excluding the label.

    Args:
        target_feature_name: The name of the target feature.
        train_df: The training data.
        corr_matrix_df: The correlation matrix of the training data.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        A tuple containing lists of OOB scores for labeled and unlabeled training data and the number of features in
        the training data.
    """
    # Prepare training data
    y_train = train_df[target_feature_name].to_numpy()

    # Remove features correlated to the target feature
    x_train = remove_features_correlated_to_target_feature(train_df, corr_matrix_df, target_feature_name, meta_data)

    oob_errors_labeled = []
    oob_errors_unlabeled = []

    # chose 1/10 of the available CPUs for nested parallel processing
    n_jobs = max(1, int(meta_data["n_cpus"] / 10))

    # Perform validation using different random seeds
    for seed in meta_data["random_seeds"]:
        assert isinstance(seed, int)
        # Create a RandomForestRegressor model with specified parameters
        clf1 = RandomForestRegressor(
            oob_score=mean_squared_error,  # Use out-of-bag error for evaluation
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=n_jobs,
        )
        # Create an exact clone of the RandomForestRegressor (clf1)
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        clf2 = clone(clf1)

        # Fit the first model with training data including the label
        clf1.fit(x_train, y_train)
        label_importance_zero = math.isclose(clf1.feature_importances_[0], 0.0)

        # If the feature importance of the label feature is zero, it means the label was not considered in the model
        if label_importance_zero:
            return None, None, x_train.shape[1]

        # Store the OOB score for the labeled model
        oob_errors_labeled.append(clf1.oob_score_)

        # Fit the second model to the unlabeled training data (excluding 'label' column)
        unlabeled_x_train = x_train.loc[:, x_train.columns != "label"]
        clf2.fit(unlabeled_x_train, y_train)

        # Store the OOB score for the unlabeled model
        oob_errors_unlabeled.append(clf2.oob_score_)
    return oob_errors_labeled, oob_errors_unlabeled, x_train.shape[1]


def validate_and_initialize_meta_data(meta_data: dict | None) -> dict:
    """Validate and initialize the metadata dictionary.

    This function ensures that the metadata dictionary contains the required keys with valid values.
    If the metadata is not provided or is missing any required keys, default values are assigned.

    Args:
        meta_data: The metadata dictionary to validate and initialize. If None, a new dictionary
            with default values is created.

    Returns:
        The validated and initialized metadata dictionary.

    Raises:
        AssertionError: If any of the required keys in the metadata dictionary have invalid types or values.

    Default Values:
        - "n_cpus": The number of available CPUs (`multiprocessing.cpu_count()`).
        - "random_seeds": A list of 30 random integers between 1 and 10000.
        - "train_correlation_threshold": A float value of 0.7, representing the absolute correlation threshold.
    """
    # initalize numpy random number generator
    rng = np.random.default_rng()

    # Set default values if meta_data is not defined
    if meta_data is None:
        meta_data = {
            # Use all available CPUs
            "n_cpus": multiprocessing.cpu_count(),
            # Generate a list of 30 random integers to initialize random forests multiple times
            "random_seeds": rng.integers(1, 10001, size=30).tolist(),
            # Absolute correlation threshold for removing features correlated to the target feature
            "train_correlation_threshold": 0.7,
        }
        logger.info(
            f"\nMeta data have been set to the default values of \n"
            f"1. {meta_data['n_cpus']} CPUs for parallel calculation \n"
            f"2. 30 random integers as seeds for random forests \n"
            f"3. 0.7 as absolute correlation threshold for removing features from the training data correlated to the "
            f"target feature"
        )
    # Ensure "n_cpus" is defined, otherwise set to the number of available CPUs
    if "n_cpus" not in meta_data:
        meta_data["n_cpus"] = multiprocessing.cpu_count()
        logger.info(f"Number of CPUs is set to {meta_data['n_cpus']}")
    # Ensure "random_seeds" is defined, otherwise generate a list of 30 random integers
    if "random_seeds" not in meta_data:
        meta_data["random_seeds"] = rng.integers(1, 10001, size=30).tolist()
        logger.info("Seeds for random forests have been set to the default of 30 random integers")
    # Ensure "train_correlation_threshold" is defined, otherwise set to 0.7
    if "train_correlation_threshold" not in meta_data:
        meta_data["train_correlation_threshold"] = 0.7
        logger.info("Absolute correlation threshold has been set to the default of 0.7")

    # Validate the types and values of the metadata keys
    assert isinstance(meta_data["n_cpus"], int), "Number of available CPUs is not an integer."
    assert isinstance(meta_data["random_seeds"], list), "Random seeds are not a list."
    assert all(
        isinstance(seed_random_forest, int) for seed_random_forest in meta_data["random_seeds"]
    ), "Random seeds are not integers."
    assert isinstance(meta_data["train_correlation_threshold"], float), "Correlation threshold is not a float."
    assert 0 <= meta_data["train_correlation_threshold"] <= 1, "Correlation threshold is not between 0 and 1."

    return meta_data


def select_feature_subset(
    data_df: pd.DataFrame,
    train_indices: np.ndarray,
    label_column_name: str = "label",
    meta_data: dict | None = None,
) -> pd.DataFrame:
    """Selects a subset of features based on the mean out-of-bag (OOB) errors for random forests regressors.

    Calculate the mean out-of-bag (OOB) errors for random forests regressors with different random seeds
    for training data including the label and without the label for each feature. It then selects a subset of features
    based on the Mann-Whitney U test which determines whether there is a significant difference between the
    two error distributions. The test is configured for the hypothesis that the distribution of
    labeled_error_distribution is shifted to the left of the unlabeled_error_distribution.
    The Mann-Whitney U test is used to calculate the p-value based on the Out-of-Bag (OOB) scores of the labeled
    and unlabeled error distributions. The p-value is a measure of the probability that an observed difference
    could have occurred just by random chance. The smaller the p-value, the greater the statistical evidence
    to reject the null hypothesis (conclude that both error distributions differ).

    Args:
        data_df: The training data. The data must contain the label and features.
        train_indices: Indices for the training split.
            The indices are used to select the training data from the data_df DataFrame.
        label_column_name: The name of the label column in the training data. Default is "label".
        meta_data: The metadata related to the dataset and experiment. If `meta_data` is `None`, default values for the
            required keys (`"n_cpus"`, `"random_seeds"`, and `"train_correlation_threshold"`) are used.
            #. The number of available CPUs (`"n_cpus"`) is required as an integer and defaults to
            `multiprocessing.cpu_count()`.
            #. A list of random seeds (`"random_seeds"`) is used to generate different error distributions by
            initializing repeated random forests multiple times. Define list of random seeds for reproducibility.
            Default is generating a random list of 30 seeds.
            #. The absolute correlation threshold for removing features from the training data correlated to the target
            feature (`"train_correlation_threshold"`) is a float between 0 and 1. The higher the threshold, the more
            features are deselected. The default value is set to `0.7` and should be adjusted if the results are not
            satisfactory.

    Returns:
        A pandas DateFrame with the selected features in the "feature_subset_selection" column.
        The feature_subset_selection column contains the fraction difference based on the mean of OOB score
        distributions, where the p_value is smaller or equal to 0.05. Features with values greater than 0 in this column
        are selected.

        The remaining columns provide additional information:
        * "feature_subset_selection_median": Contains the feature subset based on the median fraction difference.
        * "unlabeled_errors": Lists the OOB scores for the unlabeled training data.
        * "labeled_errors": Lists the OOB scores for the labeled training data.
        * "p_value": Contains the p-values from the Mann-Whitney U test.
        * "fraction_mean": Shows the fraction difference based on the mean of the distributions.
        * "fraction_median": Shows the fraction difference based on the median of the distributions.
        * "train_features_count": Indicates the number of uncorrelated features in the training data.

        The index of the DataFrame is the feature names.

    Raises:
        AssertionError: If the meta_data dictionary does not contain the required keys or if the values are not of
            the expected type. Also, if the training data does not contain any features or if the label column is not
            found in the training data.
        ValueError: If no features uncorrelated to the target feature are found.
    """
    assert label_column_name in data_df.columns, f"Label column {label_column_name} not found in the training data."

    if label_column_name != "label":
        # rename label column
        data_df[label_column_name] = data_df["label"]
    assert "label" in data_df.columns, "Label column not found in the training data."

    # check if feature importance calculation is possible
    if data_df.shape[1] < 2:
        raise ValueError("No features found in the training data.")

    meta_data = validate_and_initialize_meta_data(meta_data)

    # Split the data and select the training data
    train_df = data_df.iloc[train_indices, :]

    # Calculate the correlation matrix of the training data
    correlation_matrix_df = train_df.corr(method="spearman")

    # parallel version
    out = Parallel(n_jobs=meta_data["n_cpus"], verbose=-1)(
        delayed(calculate_oob_errors)(target_feature_name, train_df, correlation_matrix_df, meta_data)
        for target_feature_name in data_df.columns[1:]
    )
    p_value_list: list[float | None] = []
    fraction_list_mean: list[float | None] = []
    fraction_list_median: list[float | None] = []

    # Unpack the results from the parallel processing
    labeled_error_distribution_list, unlabeled_error_distribution_list, features_count_in_train_df = zip(
        *out, strict=True
    )

    # Calculate the p-value and fraction differences based on the mean and median of the distributions
    for labeled_error_distribution, unlabeled_error_distribution in zip(
        labeled_error_distribution_list, unlabeled_error_distribution_list, strict=True
    ):
        if labeled_error_distribution is None or unlabeled_error_distribution is None:
            # the current target feature was deselected because the feature importance of the label was zero
            p_value_list.append(None)
            fraction_list_mean.append(None)
            fraction_list_median.append(None)
            continue

        # calculate p-value based on the OOB scores and the mann-whitney u test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
        p_value = mannwhitneyu(labeled_error_distribution, unlabeled_error_distribution, alternative="less").pvalue
        p_value_list.append(p_value)

        # calculate the fraction difference of the two means of the distributions
        fraction_mean_based = (np.mean(unlabeled_error_distribution) - np.mean(labeled_error_distribution)) / np.mean(
            unlabeled_error_distribution
        )
        fraction_list_mean.append(fraction_mean_based)

        # calculate the fraction difference of the two medians of the distributions
        fraction_median_based = (
            np.median(unlabeled_error_distribution) - np.median(labeled_error_distribution)
        ) / np.median(unlabeled_error_distribution)
        fraction_list_median.append(fraction_median_based)

    feature_subset_selection_list = []
    feature_subset_selection_list_median = []
    # extract feature selection array
    for p_value, fraction_mean, fraction_median in zip(
        p_value_list, fraction_list_mean, fraction_list_median, strict=True
    ):
        # check if the value of the "p_value" column is smaller or equal than 0.05
        if p_value is not None and p_value <= 0.05:
            feature_subset_selection_list.append(fraction_mean)
            feature_subset_selection_list_median.append(fraction_median)
        else:
            feature_subset_selection_list.append(0.0)
            feature_subset_selection_list_median.append(0.0)

    assert (
        len(labeled_error_distribution_list)
        == len(unlabeled_error_distribution_list)
        == len(p_value_list)
        == len(fraction_list_mean)
        == len(fraction_list_median)
        == len(feature_subset_selection_list)
        == len(feature_subset_selection_list_median)
        == data_df.shape[1] - 1  # exclude label column
    )
    result_df = pd.DataFrame(index=data_df.columns[1:])
    result_df["feature_subset_selection"] = feature_subset_selection_list
    result_df["feature_subset_selection_median"] = feature_subset_selection_list_median
    result_df["unlabeled_errors"] = unlabeled_error_distribution_list
    result_df["labeled_errors"] = labeled_error_distribution_list
    result_df["p_value"] = p_value_list
    result_df["fraction_mean"] = fraction_list_mean
    result_df["fraction_median"] = fraction_list_median
    result_df["train_features_count"] = np.full(len(feature_subset_selection_list), features_count_in_train_df)
    return result_df
