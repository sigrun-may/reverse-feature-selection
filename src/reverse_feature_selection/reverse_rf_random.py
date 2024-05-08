# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Reverse feature selection with random forest regressors."""

import logging
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import preprocessing
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)


def calculate_oob_errors(
    target_feature_name: str,
    train_df: pd.DataFrame,
    corr_matrix_df: pd.DataFrame,
    meta_data: dict,
) -> Tuple[Optional[list], Optional[list]]:
    """
    Calculate out-of-bag (OOB) error for labeled and unlabeled training data.

    Args:
        target_feature_name: The name of the target feature.
        train_df: The training data.
        corr_matrix_df: The correlation matrix of the training data.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        A tuple containing lists of OOB scores for labeled and unlabeled training data.
    """
    # import omp_thread_count
    #
    # n_threads = omp_thread_count.get_thread_count()
    # print(f"Number of threads: {n_threads}")

    # Prepare training data
    assert target_feature_name in train_df.columns
    y_train = train_df[target_feature_name].to_numpy()

    # Remove features correlated to the target feature
    x_train = preprocessing.remove_features_correlated_to_target_feature(
        train_df, corr_matrix_df, target_feature_name, meta_data
    )

    oob_errors_labeled = []
    oob_errors_unlabeled = []

    # Perform validation using different random seeds
    for seed in meta_data["random_seeds"]:
        assert isinstance(seed, int)
        # Create a RandomForestRegressor model with specified parameters
        clf1 = RandomForestRegressor(
            oob_score=mean_squared_error,  # Use out-of-bag error for evaluation
            # criterion="absolute_error",
            n_estimators=100,
            random_state=seed,
            min_samples_leaf=2,
            n_jobs=30,
        )
        # Create an exact clone of the RandomForestRegressor (clf1)
        # https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html
        clf2 = clone(clf1)

        # Fit the first model with training data including the label
        assert "label" in x_train.columns
        clf1.fit(x_train, y_train)
        label_importance_zero = math.isclose(clf1.feature_importances_[0], 0.0)

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

    assert len(oob_errors_labeled) == len(oob_errors_unlabeled) == len(meta_data["random_seeds"])
    # print("oob_errors_labeled", oob_errors_labeled, "oob_errors_unlabeled", oob_errors_unlabeled)
    return oob_errors_labeled, oob_errors_unlabeled


def calculate_oob_errors_for_each_feature(
    data_df: pd.DataFrame, meta_data: dict, fold_index: int, train_indices: np.ndarray
):
    """
    Calculate the mean out-of-bag (OOB) errors for random forest regressors with different random seeds
    for training data including the label and without the label for each feature.

    Args:
        data_df: The training data.
        meta_data: The metadata related to the dataset and experiment.
        fold_index: The current loop iteration of the outer cross-validation.
        train_indices: Indices for the training split.

    Returns:
        A DataFrame containing lists of OOB scores for repeated analyzes of labeled and unlabeled data.
    """
    # Preprocess the data and cache the results if not available yet
    corr_matrix_df, train_df = preprocessing.preprocess_data(
        train_indices,
        data_df,
        fold_index,
        meta_data,
    )

    # # serial version for debugging
    # scores_labeled_list = []
    # scores_unlabeled_list = []
    # for target_feature_name in data_df.columns[1:]:
    #     scores_labeled, scores_unlabeled = calculate_oob_errors(target_feature_name, train_df, corr_matrix_df, meta_data)
    #     scores_labeled_list.append(scores_labeled)
    #     scores_unlabeled_list.append(scores_unlabeled)

    # parallel version
    out = Parallel(n_jobs=meta_data["n_cpus"], verbose=-1)(
        delayed(calculate_oob_errors)(target_feature_name, train_df, corr_matrix_df, meta_data)
        for target_feature_name in data_df.columns[1:]
    )
    scores_labeled_list = []
    scores_unlabeled_list = []
    for scores_labeled, scores_unlabeled in out:
        scores_labeled_list.append(scores_labeled)
        scores_unlabeled_list.append(scores_unlabeled)

    assert len(scores_labeled_list) == len(scores_unlabeled_list) == data_df.shape[1] - 1  # exclude label column
    result_df = pd.DataFrame(index=data_df.columns[1:])
    result_df["unlabeled"] = scores_unlabeled_list
    result_df["labeled"] = scores_labeled_list
    return result_df
