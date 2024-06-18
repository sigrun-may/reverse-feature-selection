# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Cross-validation tools."""

import logging
from datetime import datetime
from typing import Tuple

import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold


# Set up logging
logging.basicConfig(level=logging.INFO)


def cross_validate(data_df: pd.DataFrame, meta_data: dict, feature_selection_function) -> list[pd.DataFrame]:
    """Perform outer cross-validation and calculate raw values for determining feature subsets.

    Args:
        data_df: The dataset for feature selection.
        meta_data: The metadata related to the dataset and experiment.
        feature_selection_function: The function to use for feature selection.

    Returns:
        Results of the cross-validation.
    """
    start_time = datetime.utcnow()
    cv_result_list = []
    loo = LeaveOneOut()
    for fold_index, (train_indices, test_index) in enumerate(loo.split(data_df.iloc[:, 1:])):
        logging.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")
        # Calculate raw values for calculating feature subsets
        cv_result_list.append(feature_selection_function(data_df, train_indices, meta_data))
    end_time = datetime.utcnow()
    print("Duration of the cross-validation: ", end_time - start_time)
    return cv_result_list


# def loo_cross_validation(data_df: pd.DataFrame, meta_data: dict) -> Tuple[list, list]:
#     """
#     Perform leave one out cross-validation on the dataset.
#
#     Args:
#         data_df: The dataset for feature selection.
#         meta_data: The metadata related to the dataset and experiment.
#
#     Returns:
#         Results of the leave one out cross-validation and a list of indices used in cross-validation.
#     """
#     cv_result_list = []
#     cv_indices_list = []
#     loo = LeaveOneOut()
#     for fold_index, (train_indices, test_index) in enumerate(loo.split(data_df.iloc[:, 1:])):
#         logging.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")
#
#         # Calculate raw values for calculating feature subsets
#         feature_selection_instance = FeatureSelection(data_df, meta_data)
#         cv_result_list.append(feature_selection_instance.select_feature_subsets(train_indices, fold_index))
#         # Store the indices used in cross-validation for later subset validation
#         cv_indices_list.append((train_indices, test_index))
#     assert len(cv_result_list) == data_df.shape[0]
#     return cv_result_list, cv_indices_list
#
#
# def stratified_kfold_cross_validation(data_df: pd.DataFrame, meta_data: dict) -> Tuple[list, list]:
#     """
#     Perform stratified KFold cross-validation on the dataset.
#
#     Args:
#         data_df: The dataset for feature selection.
#         meta_data: The metadata related to the dataset and experiment.
#
#     Returns:
#         Results of the stratified KFold cross-validation and a list of indices used in cross-validation.
#     """
#     assert isinstance(meta_data["cv"]["n_splits"], int)
#     assert meta_data["cv"]["n_splits"] <= int(data_df.shape[0] / 2)
#     assert meta_data["cv"]["n_splits"] > 1
#
#     cv_result_list = []
#     cv_indices_list = []
#     k_fold_cv = StratifiedKFold(n_splits=meta_data["cv"]["n_splits"], shuffle=True, random_state=42)
#     for fold_index, (train_indices, test_index) in enumerate(k_fold_cv.split(data_df.iloc[:, 1:], data_df.iloc[:, 0])):
#         logging.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")
#
#         # Calculate raw values for calculating feature subsets
#         feature_selection_instance = FeatureSelection(data_df, meta_data)
#         cv_result_list.append(feature_selection_instance.select_feature_subsets(train_indices, fold_index))
#         # Store the indices used in cross-validation for later subset validation
#         cv_indices_list.append((train_indices, test_index))
#     assert len(cv_result_list) == meta_data["cv"]["n_splits"]
#     return cv_result_list, cv_indices_list
#
#
# def cross_validate_with_existing_indices(data_df: pd.DataFrame, meta_data: dict, cv_indices_list: list) -> list:
#     """
#     Perform cross-validation on the dataset using existing indices.
#
#     Args:
#         data_df: The dataset for feature selection.
#         meta_data: The metadata related to the dataset and experiment.
#         cv_indices_list: The train/ test indices to apply for cross-validation.
#     Returns:
#         Results of the cross-validation.
#     """
#     cv_result_list = []
#     for fold_index, (train_indices, test_index) in enumerate(cv_indices_list):
#         logging.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")
#
#         # Calculate raw values for calculating feature subsets
#         feature_selection_instance = FeatureSelection(data_df, meta_data)
#         cv_result_list.append(feature_selection_instance.select_feature_subsets(train_indices, fold_index))
#     return cv_result_list
