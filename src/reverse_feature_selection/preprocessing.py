# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""
This module provides functions for preprocessing data in a machine learning pipeline.

It includes functions for loading and caching preprocessed data, splitting data for training and validation,
calculating correlation matrices, removing features correlated to a target feature.

Functions:
    load_cached_data(pickle_base_path: Path, correlation_matrix: bool) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        Load the cached preprocessed data.

    preprocess_data(train_indices: np.ndarray, validation_indices: np.ndarray, data_df: pd.DataFrame, fold_index: int, meta_data: dict, correlation_matrix: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        Split data for training and validation, calculate correlation matrix if selected and cache it.

    remove_features_correlated_to_target_feature(train_df: pd.DataFrame, correlation_matrix_df: pd.DataFrame, target_feature: str, meta_data: dict) -> pd.DataFrame:
        Remove features from the training data that are correlated to the target feature.
"""


import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def preprocess_data(
    train_index: np.ndarray,
    data_df: pd.DataFrame,
    fold_index: int,
    meta_data: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data for training and validation, calculate Spearman correlation matrix and cache it.

    Args:
        train_index: Indices for the training split.
        data_df: Complete input data as a Pandas DataFrame.
        fold_index: The current loop iteration of the outer cross-validation.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        A tuple containing the validation data, training data, and the training correlation matrix.
    """
    # Check for missing values in the input DataFrame
    assert not data_df.isnull().values.any(), "Missing values detected in the input data."

    # Extract the training data based on the provided indices
    assert "label" in data_df.columns
    train_df = data_df.iloc[train_index, :]
    assert train_df.shape == (len(train_index), data_df.shape[1]), "Training data shape mismatch."

    # Scale the training data
    if meta_data["data"]["scale"]:
        robust_scaler = RobustScaler()
        train_df.iloc[:, 1:] = robust_scaler.fit_transform(train_df.iloc[:, 1:])

    if meta_data["cv"]["pickle_preprocessed_data"]:
        # Create a base path for caching the preprocessed data
        pickle_base_path = Path(f"./preprocessed_data/{meta_data['data']['name']}/outer_fold_{fold_index}")
        pickle_path = Path(f"{pickle_base_path}/train_correlation_matrix.pkl")
        # Check if the preprocessed data is already cached
        if pickle_path.exists():
            # Load the cached preprocessed data
            with open(pickle_path, "rb") as file:
                return pickle.load(file), train_df

        # Create directory for caching the preprocessed data
        pickle_base_path.mkdir(parents=True, exist_ok=True)
        assert pickle_base_path.exists()

    # Calculate the Spearman correlation matrix for the training data
    unlabeled_train_df = train_df.loc[:, train_df.columns != "label"]  # Exclude the label column
    assert "label" not in unlabeled_train_df.columns, "Label column found in unlabeled training data."
    assert unlabeled_train_df.shape[0] == len(train_index)
    assert unlabeled_train_df.shape[1] == data_df.shape[1] - 1  # Exclude the label

    train_correlation_matrix = unlabeled_train_df.corr(method="spearman")
    assert train_correlation_matrix.shape[0] == train_correlation_matrix.shape[1] == unlabeled_train_df.shape[1]
    # exclude label column from train_df
    assert (
        train_df.shape[1] - 1 == train_correlation_matrix.shape[0]
    ), f"{train_df.shape[1]} - 1 == {train_correlation_matrix.shape[1]}"
    if meta_data["cv"]["pickle_preprocessed_data"]:
        with open(f"{pickle_base_path}/train_correlation_matrix.pkl", "wb") as file:
            pickle.dump(train_correlation_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
    return train_correlation_matrix, train_df


def remove_features_correlated_to_target_feature(
    train_df: pd.DataFrame, correlation_matrix_df: pd.DataFrame, target_feature: str, meta_data: dict
) -> pd.DataFrame:
    """
    Remove features from the training data that are correlated to the target feature.

    This function creates a mask for uncorrelated features based on the correlation threshold specified in the metadata.
    It then uses this mask to select the uncorrelated features from the training data.

    Args:
        train_df: The training data.
        correlation_matrix_df: The correlation matrix of the training data.
        target_feature: The name of the target feature.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        The training data with only the uncorrelated features remaining.
    """
    # # check if the target feature is in the training data
    # assert target_feature in train_df.columns
    #
    # # check if the target feature is in the correlation matrix
    # assert target_feature in correlation_matrix_df.index
    #
    # # Extract the unlabeled training data
    # assert train_df.shape[1] - 1 == correlation_matrix_df.shape[1]  # Exclude the label
    # assert train_df.columns[0] == "label"
    unlabeled_train_df = train_df.loc[:, train_df.columns != "label"]

    # assert unlabeled_train_df.shape[1] == correlation_matrix_df.shape[0] == correlation_matrix_df.shape[1]

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
    # assert uncorrelated_features_mask.shape[0] == unlabeled_train_df.shape[1]

    # Get the column names of uncorrelated features
    uncorrelated_feature_names = unlabeled_train_df.columns[uncorrelated_features_mask]

    # # Ensure that "label" is not in the list of uncorrelated features
    # assert "label" not in uncorrelated_feature_names
    #
    # # Check if there aren't any features left after removing "label" and "target_feature"
    # if uncorrelated_feature_names.size == 0:
    #     raise ValueError(
    #         f"No features uncorrelated to {target_feature} with absolute correlation threshold "
    #         f"{meta_data['train_correlation_threshold']}"
    #     )

    # # Check if the maximum correlation of any uncorrelated feature with the target is within the threshold
    # assert (
    #     correlation_matrix_df.loc[target_feature, uncorrelated_features_mask].abs().max()
    #     <= meta_data["train_correlation_threshold"]
    # ), f"{meta_data['train_correlation_threshold']}"

    # Insert the 'label' as the first column
    uncorrelated_feature_names = uncorrelated_feature_names.insert(0, "label")

    # Remove correlated features from the training data
    uncorrlated_train_df = train_df[uncorrelated_feature_names]
    # assert target_feature not in uncorrlated_train_df.columns
    # assert "label" == uncorrlated_train_df.columns[0], f"{uncorrlated_train_df.columns[0]}"
    # assert uncorrlated_train_df.shape[0] == train_df.shape[0]
    # assert uncorrlated_train_df.shape[1] == uncorrelated_feature_names.size

    # Return the data frame with uncorrelated features
    return uncorrlated_train_df
