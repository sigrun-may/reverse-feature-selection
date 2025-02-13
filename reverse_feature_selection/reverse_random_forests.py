# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Reverse feature selection with random forest regressors."""

import logging
import math

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from reverse_feature_selection import preprocessing

logging.basicConfig(level=logging.INFO)


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
        A tuple containing lists of OOB scores for labeled and unlabeled training data.
    """
    # Prepare training data
    y_train = train_df[target_feature_name].to_numpy()

    # Remove features correlated to the target feature
    x_train = preprocessing.remove_features_correlated_to_target_feature(
        train_df, corr_matrix_df, target_feature_name, meta_data
    )

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


def select_feature_subset(data_df: pd.DataFrame, train_indices: np.ndarray, meta_data: dict) -> pd.DataFrame:
    """Selects a subset of features based on the mean out-of-bag (OOB) errors for random forest regressors.

    Calculate the mean out-of-bag (OOB) errors for random forest regressors with different random seeds
    for training data including the label and without the label for each feature. It then selects a subset of features
    based on the Mann-Whitney U test which determines whether there is a significant difference between the
    two error distributions. The test is configured for the hypothesis that the distribution of
    labeled_error_distribution is shifted to the left of the unlabeled_error_distribution.
    The Mann-Whitney U test is used to calculate the p-value based on the Out-of-Bag (OOB) scores of the labeled
    and unlabeled error distributions. The p-value is a measure of the probability that an observed difference
    could have occurred just by random chance. The smaller the p-value, the greater the statistical evidence
    to reject the null hypothesis (conclude that both error distributions differ).

    Args:
        data_df: The training data. The first column must contain the label and the remaining columns the features.
            The label column must be named "label".
        train_indices: Indices for the training split.
        meta_data: The metadata related to the dataset and experiment.
            The required keys are: "n_cpus", "random_seeds" and "train_correlation_threshold". Number of available CPUs
            as integer and a list of random seeds for reproducibility of the repeated reverse random forest.
            The correlation threshold for removing correlated features is a float between 0 and 1. The threshold is used
            to remove features correlated to the target feature. The target feature is the feature for which the
            feature subset is selected. The higher the threshold, the more features are removed.

    Returns:
        A DataFrame containing raw data with lists of OOB scores for repeated analyzes of labeled and unlabeled data,
        p_values and fraction differences based on means and medians of the distributions. The feature_subset_selection
        column contains the fraction difference based on the mean of the distributions, where the p_value is smaller or
        equal to 0.05. Those features are selected for the feature subset.
    """
    assert "n_cpus" in meta_data, "Number of available CPUs not found in meta_data."
    assert "random_seeds" in meta_data, "Random seeds not found in meta_data."
    assert isinstance(meta_data["n_cpus"], int), "Number of available CPUs is not an integer."
    assert isinstance(meta_data["random_seeds"], list), "Random seeds are not a list."
    assert all(
        isinstance(seed_random_forest, int) for seed_random_forest in meta_data["random_seeds"]
    ), "Random seeds are not integers."
    assert "train_correlation_threshold" in meta_data, "Correlation threshold not found in meta_data."
    assert isinstance(meta_data["train_correlation_threshold"], float), "Correlation threshold is not a float."
    assert 0 <= meta_data["train_correlation_threshold"] <= 1, "Correlation threshold is not between 0 and 1."
    assert "label" in data_df.columns, "Label column not found in the training data."
    assert data_df.shape[1] > 1, "No features found in the training data."

    # check if feature importance calculation is possible
    if data_df.shape[1] < 2:
        raise ValueError("No features found in the training data.")

    # Split the training data into a training and test set
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
