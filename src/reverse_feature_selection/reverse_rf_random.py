import multiprocessing
import pickle
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import ravel
from scipy.stats import ttest_ind
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
)

from src.reverse_feature_selection import preprocessing


def calculate_oob_errors(x_train: pd.DataFrame, y_train: np.ndarray):
    """
        Calculate out-of-bag (OOB) error for labeled and unlabeled training data.

        Args:
            x_train: The training data.
            y_train: The target values for the training data.

        Returns:
            A tuple containing lists of OOB scores for labeled and unlabeled data.
        """
    ...

    oob_errors_labeled = []
    oob_errors_unlabeled = []

    # Perform validation using different random seeds
    for i in range(5):
        # Create a RandomForestRegressor model with specified parameters
        clf1 = RandomForestRegressor(
            warm_start=False,
            max_features=None,
            oob_score=mean_squared_error,  # Use out-of-bag score for evaluation
            n_estimators=100,
            # random_state=seed,
            min_samples_leaf=2,
        )
        # Create a copy of the RandomForestRegressor (clf1)
        clf2 = clone(clf1)

        # Fit the first model with training data including the label
        assert "label" in x_train.columns
        clf1.fit(x_train, y_train)
        label_importance_zero = clf1.feature_importances_[0] == 0

        # If the feature importance of the label feature is zero, it means the label was not considered in the model
        if label_importance_zero:
            return None, None

        # Store the OOB score for the labeled model
        oob_errors_labeled.append(clf1.oob_score_)

        # Fit the second model to the unlabeled training data (excluding 'label' column)
        unlabeled_x_train = x_train.loc[:, x_train.columns != "label"]
        assert unlabeled_x_train.shape[1] == x_train.shape[1] - 1
        assert "label" not in unlabeled_x_train.columns
        clf2.fit(unlabeled_x_train, y_train)

        # Store the OOB score for the unlabeled model
        oob_errors_unlabeled.append(clf2.oob_score_)

    # print("oob_errors_labeled", oob_errors_labeled, "oob_errors_unlabeled", oob_errors_unlabeled)
    return oob_errors_labeled, oob_errors_unlabeled


def calculate_mean_oob_errors_and_p_value(target_feature_name, outer_cv_loop, meta_data):
    """
    Calculate the mean out-of-bag (OOB) errors for random forest regressors with different random seeds
    for training data including the label and without the label for the given target feature.

    Args:
        target_feature_name: The name of the target feature.
        outer_cv_loop: The current loop iteration of the outer cross-validation.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        tuple: A tuple containing the mean OOB score for labeled data, the mean OOB score for unlabeled data, and the p-value.
    """

    # calculate the mean oob_scores for random forest regressors with different random seeds
    # for training data including the label and without the label for the given target feature

    mean_oob_score_labeled = 0
    mean_oob_score_unlabeled = 0
    p_value = None

    pickle_base_path = Path(f"../../preprocessed_data/{meta_data['data']['name']}/outer_fold_{outer_cv_loop}")
    assert pickle_base_path.exists(), f"{pickle_base_path} does not exist"

    # Load the cached preprocessed data for the given outer cross-validation fold
    with open(f"{pickle_base_path}/train.pkl", "rb") as file:
        train_df = pickle.load(file)
        assert target_feature_name in train_df.columns
        assert "label" in train_df.columns
    with open(f"{pickle_base_path}/train_correlation_matrix.pkl", "rb") as file:
        corr_matrix_df = pickle.load(file)
        assert "label" not in corr_matrix_df.columns
    assert train_df.shape[1] - 1 == corr_matrix_df.shape[0]  # corr_matrix_df does not include the label

    # Prepare training data
    y_train = ravel(train_df[target_feature_name])

    # Remove features correlated to the target feature
    x_train = preprocessing.remove_features_correlated_to_target_feature(
        train_df, corr_matrix_df, target_feature_name, meta_data
    )
    if x_train is None:
        return None, None, None

    assert target_feature_name not in x_train.columns

    # Calculate out-of-bag (OOB) scores for labeled and unlabeled training data
    oob_scores_labeled, oob_scores_unlabeled = calculate_oob_errors(x_train, y_train)

    # Check if OOB scores for labeled data are available and if training with the label is better than without the label
    if oob_scores_labeled is not None and abs(np.mean(oob_scores_labeled)) < abs(np.mean(oob_scores_unlabeled)):
        # Calculate the percentage difference between OOB scores
        absolute_percentage_difference = (
            (np.mean(oob_scores_unlabeled) - np.mean(oob_scores_labeled)) / abs(np.mean(oob_scores_unlabeled))
        ) * 100
        if abs(absolute_percentage_difference) >= 5:
            print("percentage_difference", absolute_percentage_difference)

        # Perform the t-test (welsh)
        p_value = ttest_ind(oob_scores_labeled, oob_scores_unlabeled, alternative="less", equal_var=False).pvalue

        # Perform the mannwhitneyu test
        # p_value = mannwhitneyu(oob_scores_labeled, oob_scores_unlabeled, alternative="less").pvalue

        # Check if the result is statistically significant (alpha level = 0.05)
        if p_value <= 0.05:
            mean_oob_score_labeled = np.mean(oob_scores_labeled)
            mean_oob_score_unlabeled = np.mean(oob_scores_unlabeled)
            print(f"p_value {target_feature_name} {p_value} l: {mean_oob_score_labeled} ul: {mean_oob_score_unlabeled}")

            # Calculate the percentage difference between OOB scores
            absolute_percentage_difference = (
                abs((mean_oob_score_unlabeled - mean_oob_score_labeled) / abs(mean_oob_score_unlabeled)) * 100
            )
            print("absolute_percentage_difference", absolute_percentage_difference)

            # Calculate a metric based on the percentage difference and p-value
            print("metric", absolute_percentage_difference / log(p_value))
            print("---------")

    return mean_oob_score_labeled, mean_oob_score_unlabeled, p_value


def calculate_oob_errors_per_feature(data_df, meta_data, fold_index):
    scores_labeled_list = []
    scores_unlabeled_list = []
    p_values_list = []

    # # serial version for debugging
    # for target_feature_name in data_df.columns[1:]:
    #     score_labeled, score_unlabeled, p_value = calculate_mean_oob_scores_and_p_value(target_feature_name, fold_index, meta_data)
    #     scores_labeled_list.append(score_labeled)
    #     scores_unlabeled_list.append(score_unlabeled)
    #     p_values_list.append(p_value)

    # parallel version
    out = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=-1)(
        delayed(calculate_mean_oob_errors_and_p_value)(target_feature_name, fold_index, meta_data)
        for target_feature_name in data_df.columns[1:]
    )

    for score_labeled, score_unlabeled, p_value in out:
        scores_labeled_list.append(score_labeled)
        scores_unlabeled_list.append(score_unlabeled)
        p_values_list.append(p_value)

    assert len(scores_labeled_list) == len(scores_unlabeled_list) == data_df.shape[1] - 1  # exclude label column
    result_df = pd.DataFrame(data=scores_unlabeled_list, index=data_df.columns[1:], columns=["unlabeled"])
    result_df["labeled"] = scores_labeled_list
    result_df["p_values"] = p_values_list
    return result_df
