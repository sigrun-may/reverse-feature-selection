# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""This preprocessing module removes features from the training data that are correlated to the target feature."""


import pandas as pd


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
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        The training data including the label with only the features uncorrelated to the target feature remaining .
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
    uncorrlated_train_df = train_df[train_df.columns[uncorrelated_features_mask]]

    assert len(uncorrlated_train_df.columns) > 1, "No features uncorrelated to the target feature found."

    # insert the 'label' as the first column if it is not already there
    if uncorrlated_train_df.columns[0] != "label":
        uncorrlated_train_df.insert(0, "label", train_df["label"])

    # Return the data frame with uncorrelated features
    return uncorrlated_train_df
