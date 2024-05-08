# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Cross-validation tools."""

import logging
from typing import Tuple

import pandas as pd
from feature_selection import FeatureSelection
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

# Set up logging
logging.basicConfig(level=logging.INFO)


def cross_validate(data_df: pd.DataFrame, meta_data: dict, result_dict: dict) -> dict:
    """
    Perform outer cross-validation and calculate raw values for determining feature subsets.

    Args:
        data_df: The dataset for feature selection.
        meta_data: The metadata related to the dataset and experiment.
        result_dict: The dictionary to store the results of the cross-validation. If empty, new cross-validation indices
            are created. Else, the existing cross-validation indices of the given dictionary are used.

    Returns:
        Results of the cross-validation.
    """
    assert isinstance(meta_data["cv"]["loo"], bool)
    if meta_data["cv"]["loo"] is None:
        raise ValueError("Invalid cross-validation method.")

    # check if result_dict is empty
    if not result_dict:
        # Create new cross-validation indices
        cv_result_list, cv_indices_list = (
            loo_cross_validation(data_df, meta_data)
            if meta_data["cv"]["loo"]
            else stratified_kfold_cross_validation(data_df, meta_data)
        )
        result_dict = {
            meta_data["method"]: cv_result_list,
            "indices": cv_indices_list,
        }
    else:
        # Use existing cross-validation indices from given result_dict
        cv_result_list = cross_validate_with_existing_indices(data_df, meta_data, result_dict["indices"])
        result_dict[meta_data["method"]] = cv_result_list
    assert len(result_dict["indices"]) == len(result_dict[meta_data["method"]])
    return result_dict


def loo_cross_validation(data_df: pd.DataFrame, meta_data: dict) -> Tuple[list, list]:
    """
    Perform leave one out cross-validation on the dataset.

    Args:
        data_df: The dataset for feature selection.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        Results of the leave one out cross-validation and a list of indices used in cross-validation.
    """
    cv_result_list = []
    cv_indices_list = []
    loo = LeaveOneOut()
    for fold_index, (train_indices, test_index) in enumerate(loo.split(data_df.iloc[:, 1:])):
        logging.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")

        # Calculate raw values for calculating feature subsets
        feature_selection_instance = FeatureSelection(data_df, meta_data)
        cv_result_list.append(feature_selection_instance.select_feature_subsets(train_indices, fold_index))
        # Store the indices used in cross-validation for later subset validation
        cv_indices_list.append((train_indices, test_index))
    assert len(cv_result_list) == data_df.shape[0]
    return cv_result_list, cv_indices_list


def stratified_kfold_cross_validation(data_df: pd.DataFrame, meta_data: dict) -> Tuple[list, list]:
    """
    Perform stratified KFold cross-validation on the dataset.

    Args:
        data_df: The dataset for feature selection.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        Results of the stratified KFold cross-validation and a list of indices used in cross-validation.
    """
    assert isinstance(meta_data["cv"]["n_splits"], int)
    assert meta_data["cv"]["n_splits"] <= int(data_df.shape[0] / 2)
    assert meta_data["cv"]["n_splits"] > 1

    cv_result_list = []
    cv_indices_list = []
    k_fold_cv = StratifiedKFold(n_splits=meta_data["cv"]["n_splits"], shuffle=True, random_state=42)
    for fold_index, (train_indices, test_index) in enumerate(k_fold_cv.split(data_df.iloc[:, 1:], data_df.iloc[:, 0])):
        logging.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")

        # Calculate raw values for calculating feature subsets
        feature_selection_instance = FeatureSelection(data_df, meta_data)
        cv_result_list.append(feature_selection_instance.select_feature_subsets(train_indices, fold_index))
        # Store the indices used in cross-validation for later subset validation
        cv_indices_list.append((train_indices, test_index))
    assert len(cv_result_list) == meta_data["cv"]["n_splits"]
    return cv_result_list, cv_indices_list


def cross_validate_with_existing_indices(data_df: pd.DataFrame, meta_data: dict, cv_indices_list: list) -> list:
    """
    Perform cross-validation on the dataset using existing indices.

    Args:
        data_df: The dataset for feature selection.
        meta_data: The metadata related to the dataset and experiment.
        cv_indices_list: The train/ test indices to apply for cross-validation.
    Returns:
        Results of the cross-validation.
    """
    cv_result_list = []
    for fold_index, (train_indices, test_index) in enumerate(cv_indices_list):
        logging.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")

        # Calculate raw values for calculating feature subsets
        feature_selection_instance = FeatureSelection(data_df, meta_data)
        cv_result_list.append(feature_selection_instance.select_feature_subsets(train_indices, fold_index))
    return cv_result_list
