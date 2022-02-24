from typing import Dict, Tuple, List, Union

import joblib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, Memory
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import load_breast_cancer

import settings
import utils
import weighted_knn
from preprocessing import (
    cluster_data,
    yeo_johnson_transform_test_train_splits,
    parse_data,
)
import filter_methods
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
    meta_data = settings.get_meta_data()
    data_df = parse_data(meta_data)
    # print(f"perfect features: '{utils.get_well_separated_data(data_df)}")
    # scores, min, max = filter_methods.get_scores(data_df.iloc[:, :50])
    # print(utils.sort_list_of_tuples_by_index(scores))
    # print(f"good features (max, min): "
    #       f"'{filter_methods.get_scores(data_df.iloc[:, :50])}")
    if meta_data["data"]["cluster_correlation_threshold"]:
        data_df, cluster_dict = cluster_data(data_df, meta_data)
        print("cluster shape", data_df.shape)

    # clustered_data_df2, cluster_dict2 = cluster_data(clustered_data_df)
    # print("cluster shape 2", clustered_data_df2.shape)
    meta_data["data"]["columns"] = data_df.columns.tolist()
    settings.save_meta_data(meta_data)

    # select feature subsets
    feature_selection_result = load_or_generate_feature_subsets(data_df, meta_data)
    (
        feature_selection_result_dict,
        test_train_indices_list,
    ) = _extract_indices_and_results(feature_selection_result)

    # validate feature subsets
    metrics_per_method_dict = evaluate_feature_selection_methods(
        feature_selection_result_dict,
        test_train_indices_list,
        data_df,
        k_neighbors=5,
    )
