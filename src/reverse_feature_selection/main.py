from typing import Dict, Tuple, List, Union

import joblib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut

import settings
import weighted_knn
from preprocessing import cluster_data, yeo_johnson_transform_test_train_splits
import standard_lasso_feature_selection

def parse_data(number_of_features: int, path: str) -> pd.DataFrame:
    data_df = pd.read_csv(path)
    data_df = data_df.iloc[:, :number_of_features]
    print(data_df.shape)
    return data_df


def evaluate_selected_features(
    all_feature_subsets: List[Dict[str, float]]
) -> Tuple[set, set]:
    intersection_of_features = union_of_features = set(all_feature_subsets[0].keys())
    for i in range(len(all_feature_subsets) - 1):
        feature_set = set(all_feature_subsets[i + 1].keys())
        intersection_of_features = intersection_of_features.intersection(feature_set)
        union_of_features = union_of_features.union(feature_set)
    return union_of_features, intersection_of_features


def select_feature_subset(data: pd.DataFrame, train_index, test_index, index: int):
    # load or generate transformed and standardized train test splits with respective train correlation matrix
    transformed_data_path_iteration = (
        f"{settings.DIRECTORY_FOR_PICKLED_FILES}/{settings.EXPERIMENT_NAME}/"
        f"transformed_test_train_splits_dict_{str(index)}.pkl"
    )
    preprocessed_data_dict = yeo_johnson_transform_test_train_splits(
        settings.N_FOLDS_INNER_CV, data.iloc[train_index, :], None
    )

    selected_feature_subset={
    'standard_lasso' : standard_lasso_feature_selection.select_features(preprocessed_data_dict),
    'reverse_lasso' : standard_lasso_feature_selection.select_features(preprocessed_data_dict),
    'random_forest' : standard_lasso_feature_selection.select_features(preprocessed_data_dict),
    }
    # selected_feature_subset = select_features(preprocessed_data_dict)
    # selected_feature_subset = selection_method_rf(preprocessed_data_dict)



    # # append metrics to overall result for outer cross-validation
    # for k, v in metrics_dict.items():
    #     r = validation_metrics_dict.get(k, [])
    #     r.append(v)
    #     validation_metrics_dict[k] = r

    return selected_feature_subset


if __name__ == "__main__":
    data_df = parse_data(settings.NUMBER_OF_FEATURES, settings.INPUT_DATA_PATH)
    clustered_data_df, cluster_dict = cluster_data(data_df)

    # outer cross-validation to validate selected feature subsets
    loo = LeaveOneOut()
    feature_subsets = Parallel(n_jobs=settings.N_JOBS, verbose=5)(
        delayed(select_feature_subset)(clustered_data_df, train_index, index)
        for index, (train_index, test_index) in enumerate(loo.split(clustered_data_df))
    )
    # if settings.SAVE_RESULT:
    joblib.dump(feature_subsets, settings.PATH_TO_RESULT, compress=("gzip", 3))

    for feature_subset in feature_subsets:
        metrics = Parallel(n_jobs=settings.N_JOBS, verbose=5)(
            delayed(weighted_knn.validate_feature_subset)(
                test=clustered_data_df.iloc[test_index, :],
                train=clustered_data_df.iloc[train_index, :],
                selected_feature_subset=feature_subset,
                number_of_neighbors=3,
                )
            for index, (train_index, test_index) in enumerate(loo.split(clustered_data_df))
    )

    # # outer cross-validation to validate feature subsets
    # loo = LeaveOneOut()
    # feature_subsets = []
    # validation_metrics_dict = {}
    # for index, (train_index, test_index) in enumerate(loo.split(clustered_data_df)):
    #     # load or generate transformed and standardized train test splits with respective train correlation matrix
    #     transformed_data_path_iteration = (
    #         f"{settings.DIRECTORY_FOR_PICKLED_FILES}/{settings.EXPERIMENT_NAME}/"
    #         f"transformed_test_train_splits_dict_{str(index)}.pkl"
    #     )
    #     preprocessed_data_dict = yeo_johnson_transform_test_train_splits(
    #         settings.N_FOLDS_INNER_CV, clustered_data_df.iloc[train_index, :], None
    #     )
    #
    #     selected_feature_subset = select_features(preprocessed_data_dict)
    #     # selected_feature_subset = selection_method_reverse_lasso(preprocessed_data_dict)
    #     # selected_feature_subset = selection_method_rf(preprocessed_data_dict)
    #     feature_subsets.append(selected_feature_subset)
    #
    #     metrics_dict = weighted_knn.validate_feature_subset(
    #         test=clustered_data_df.iloc[test_index, :],
    #         train=clustered_data_df[selected_feature_subset.keys()].iloc[
    #             train_index, :
    #         ],
    #         selected_feature_subset=selected_feature_subset,
    #         number_of_neighbors=3,
    #     )
    #
    #     # append metrics to overall result for outer cross-validation
    #     for k, v in metrics_dict.items():
    #         r = validation_metrics_dict.get(k, [])
    #         r.append(v)
    #         validation_metrics_dict[k] = r

    validation_metrics_dict = {
        k: np.mean(v) for k, v in validation_metrics_dict.items()
    }

    print(
        "union_of_features",
        "intersection_of_features",
        evaluate_selected_features(feature_subsets),
    )
