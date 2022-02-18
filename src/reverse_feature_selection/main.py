from typing import Dict, Tuple, List, Union

import joblib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, Memory
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import load_breast_cancer

import settings
import weighted_knn
from preprocessing import (
    cluster_data,
    yeo_johnson_transform_test_train_splits,
    parse_data,
)
import standard_lasso_feature_selection
import reverse_lasso_feature_selection
import rf_feature_selection
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    mutual_info_score,
)

from src.reverse_feature_selection.feature_subset_selection import (
    load_or_generate_feature_subsets,
)
from src.reverse_feature_selection.validation import (
    _extract_indices_and_results,
    evaluate_feature_selection_methods,
)

if __name__ == "__main__":
    input_data_df = parse_data(settings.NUMBER_OF_FEATURES, settings.INPUT_DATA_PATH)
    clustered_data_df, cluster_dict = cluster_data(input_data_df)
    print("cluster shape", clustered_data_df.shape)

    # clustered_data_df2, cluster_dict2 = cluster_data(clustered_data_df)
    # print("cluster shape 2", clustered_data_df2.shape)

    # select feature subsets
    feature_selection_result = load_or_generate_feature_subsets(clustered_data_df)
    (
        feature_selection_result_dict,
        test_train_indices_list,
    ) = _extract_indices_and_results(feature_selection_result)

    # validate feature subsets
    evaluate_feature_selection_methods(
        feature_selection_result_dict,
        test_train_indices_list,
        clustered_data_df,
        k_neighbors=7,
    )
