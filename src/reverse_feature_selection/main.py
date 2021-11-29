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
import reverse_lasso_feature_selection
import rf_feature_selection


def parse_data(number_of_features: int, path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    # data = data.iloc[:, :number_of_features]
    print(data.shape)
    return data


def evaluate_selected_features(
    all_feature_subsets: List[Dict[str, float]]
) -> Tuple[set, set]:
    intersection_of_features = union_of_features = set(all_feature_subsets[0].keys())
    for i in range(len(all_feature_subsets) - 1):
        feature_set = set(all_feature_subsets[i + 1].keys())
        intersection_of_features = intersection_of_features.intersection(feature_set)
        union_of_features = union_of_features.union(feature_set)
    return union_of_features, intersection_of_features


def select_feature_subset(
    data: pd.DataFrame, train_index, outer_cv_loop: int
):
    print("iteration: ", outer_cv_loop, " #############################################################")
    # load or generate transformed and standardized train test splits with respective train correlation matrix
    transformed_data_path_iteration = (
        f"{settings.DIRECTORY_FOR_PICKLED_FILES}/{settings.EXPERIMENT_NAME}/"
        f"transformed_test_train_splits_dict_{outer_cv_loop}.pkl"
    )
    preprocessed_data_dict = yeo_johnson_transform_test_train_splits(
        settings.N_FOLDS_INNER_CV, data.iloc[train_index, :], None
    )

    # TODO pickle feature subsets
    selected_feature_subset = {
        "standard_lasso": standard_lasso_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop
        ),
        "reverse_lasso": reverse_lasso_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop, settings.CORRELATION_THRESHOLD_REGRESSION
        ),
        "random_forest": rf_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop
        ),
    }
    # print(selected_features)
    # selected_features = select_features(preprocessed_data_dict)
    # selected_features = selection_method_rf(preprocessed_data_dict)

    # # append metrics to overall result for outer cross-validation
    # for selection_method, selected_features in metrics_dict.items():
    #     r = validation_metrics_dict.get(selection_method, [])
    #     r.append(selected_features)
    #     validation_metrics_dict[selection_method] = r

    return selected_feature_subset


if __name__ == "__main__":
    # data_df1 = parse_data(settings.NUMBER_OF_FEATURES, settings.INPUT_DATA_PATH)
    # data_df1.iloc[:, 1:].to_csv("../../data/small_50.csv", index=False)
    data_df = parse_data(settings.NUMBER_OF_FEATURES, settings.INPUT_DATA_PATH)
    clustered_data_df, cluster_dict = cluster_data(data_df)

    # outer cross-validation to validate selected feature subsets
    loo = LeaveOneOut()
    feature_subsets = Parallel(n_jobs=settings.N_JOBS, verbose=5)(
        delayed(select_feature_subset)(
            clustered_data_df,
            train_index,
            index,
        )
        for index, (train_index, test_index) in enumerate(loo.split(clustered_data_df))
    )
    # if settings.SAVE_RESULT:
    # print(feature_subsets)
    # joblib.dump(feature_subsets, settings.PATH_TO_RESULT, compress=("gzip", 3))


        # print(feature_subset)


    # for feature_subset in feature_subsets:
    #     metrics = Parallel(n_jobs=settings.N_JOBS, verbose=5)(
    #         delayed(weighted_knn.validate_feature_subset)(
    #             test=clustered_data_df.iloc[test_index, :],
    #             train=clustered_data_df.iloc[train_index, :],
    #             selected_features=feature_subset,
    #             number_of_neighbors=3,  # TODO intern optimieren
    #         )
    #         for index, (train_index, test_index) in enumerate(
    #             loo.split(clustered_data_df)
    #         )
    #     )

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
    #     selected_features = select_features(preprocessed_data_dict)
    #     # selected_features = selection_method_reverse_lasso(preprocessed_data_dict)
    #     # selected_features = selection_method_rf(preprocessed_data_dict)
    #     feature_subsets.append(selected_features)
    #
    #     metrics_dict = weighted_knn.validate_feature_subset(
    #         test=clustered_data_df.iloc[test_index, :],
    #         train=clustered_data_df[selected_features.keys()].iloc[
    #             train_index, :
    #         ],
    #         selected_features=selected_features,
    #         number_of_neighbors=3,
    #     )
    #
    #     # append metrics to overall result for outer cross-validation
    #     for selection_method, selected_features in metrics_dict.items():
    #         r = validation_metrics_dict.get(selection_method, [])
    #         r.append(selected_features)
    #         validation_metrics_dict[selection_method] = r

    # validation_metrics_dict = {
    #     selection_method: np.mean(selected_features) for selection_method, selected_features in validation_metrics_dict.items()
    # }
    # rewrite data structure
    feature_selection_result_dict = {}
    for feature_subset in feature_subsets:
        for selection_method, selected_features in feature_subset.items():
            selected_feature_subset_list = feature_selection_result_dict.get(selection_method, [])
            selected_feature_subset_list.append(selected_features)
            feature_selection_result_dict[selection_method] = selected_feature_subset_list

    for feature_selection_method, selected_feature_subsets in feature_selection_result_dict.items():
        union, intersect = evaluate_selected_features(selected_feature_subsets)
        print(feature_selection_method, ": ")
        print("union_of_features: ", union)
        print("intersection_of_features: ", intersect)
