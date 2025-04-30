# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Cross-validation tools."""
import datetime
import logging

import pandas as pd
from sklearn.model_selection import LeaveOneOut

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cross_validate(data_df: pd.DataFrame, meta_data: dict, feature_selection_function) -> list[pd.DataFrame]:
    """Perform outer cross-validation and calculate raw values for determining feature subsets.

    Args:
        data_df: The dataset for feature selection.
        meta_data: The metadata related to the dataset and experiment.
        feature_selection_function: The function to use for feature selection.

    Returns:
        Results of the cross-validation.
    """
    start_time = datetime.datetime.now(tz=datetime.timezone.utc)
    cv_result_list = []
    loo = LeaveOneOut()
    for fold_index, (train_indices, _) in enumerate(loo.split(data_df)):
        logger.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")
        # Calculate raw values for calculating feature subsets
        selected_feature_subset = feature_selection_function(data_df, train_indices, meta_data)
        cv_result_list.append(selected_feature_subset)
    end_time = datetime.datetime.now(tz=datetime.timezone.utc)
    print("Duration of the cross-validation: ", end_time - start_time)
    return cv_result_list
