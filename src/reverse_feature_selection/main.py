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

# from src.reverse_feature_selection.misc import (
#     parse_data,
#     _extract_indices_and_results,
#     validate_feature_subsets, load_or_generate_feature_subsets_old,
# )
from src.reverse_feature_selection.feature_subset_selection import (
    load_or_generate_feature_subsets_old,
)
from src.reverse_feature_selection.validation import (
    _extract_indices_and_results,
    evaluate_feature_selection_methods,
)

if __name__ == "__main__":
    # X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    # data_df = pd.DataFrame()
    # data_df["label"] = y
    # data_df = data_df.join(X)
    input_data_df = parse_data(settings.NUMBER_OF_FEATURES, settings.INPUT_DATA_PATH)
    clustered_data_df, cluster_dict = cluster_data(input_data_df)
    # # clustered_data_df.to_csv('clustered_1200.csv', index = False)
    # # clustered_data_df = pd.read_csv("clustered_1200.csv")
    # #
    # outer cross-validation to validate selected feature subsets
    # loo = LeaveOneOut()
    # fold_splitter = StratifiedKFold(n_splits=settings.N_FOLDS_OUTER_CV)
    # feature_subsets = Parallel(n_jobs=settings.N_JOBS, verbose=5)(
    #     delayed(select_feature_subset)(
    #         clustered_data_df,
    #         train_indices,
    #         test_indices,
    #         outer_cv_loop_iteration,
    #     )
    #     for outer_cv_loop_iteration, (train_indices, test_indices) in enumerate(
    #         # loo.split(clustered_data_df)
    #         fold_splitter.split(
    #             clustered_data_df.iloc[:, 1:], clustered_data_df["label"]
    #         )
    #     )
    # )
    # # if settings.SAVE_RESULT:
    # joblib.dump(feature_subsets, settings.PATH_TO_RESULT, compress=("gzip", 3))
    # feature_selection_result = load_or_generate_feature_subsets(clustered_data_df)
    feature_selection_result = load_or_generate_feature_subsets_old(clustered_data_df)

    (
        feature_selection_result_dict,
        test_train_indices_list,
    ) = _extract_indices_and_results(feature_selection_result)

    # optimize number of neighbors
    for hyperparameter_k_neighbors in [3, 5]:
        # for hyperparameter_k_neighbors in [3, 5, 7, 9, 11]:
        # min_difference = np.asarray(range(1, 7, 1)) / 10
        # for delta in min_difference:
        #     print(delta)
        print(
            evaluate_feature_selection_methods(
                feature_selection_result_dict,
                test_train_indices_list,
                clustered_data_df,
                hyperparameter_k_neighbors,
            )
        )

    # feature_subsets = parallel_feature_subset_selection(clustered_data_df)
    # feature_subsets = joblib.load(settings.PATH_TO_RESULT)
    # # outer cross-validation to validate selected feature subsets
    # loo = LeaveOneOut()
    # predicted_classes = Parallel(n_jobs=1, verbose=5)(
    #     delayed(weighted_knn.validate)(
    #         clustered_data_df.iloc[test_indices, :],
    #         clustered_data_df.iloc[train_indices, :],
    #         feature_subsets[outer_cv_loop_iteration]["random_forest"],
    #         5,
    #     )
    #     for outer_cv_loop_iteration, (train_indices, test_indices) in enumerate(
    #         loo.split(clustered_data_df)
    #     )
    # )
    # print(predicted_classes)
    # true_classes = list(clustered_data_df["label"])
    # print(
    #     "matthews",
    #     matthews_corrcoef(true_classes, predicted_classes),
    #     "accuracy",
    #     accuracy_score(true_classes, predicted_classes),
    #     "f1_score",
    #     f1_score(true_classes, predicted_classes),
    #     "balanced_accuracy_score",
    #     balanced_accuracy_score(true_classes, predicted_classes),
    # )
    #  # Only use the labels that appear in the data
    # classes = unique_labels(true_classes, predicted_classes)
    # plotter.plot_confusion_matrix(true_classes, predicted_classes, classes, True, title=None, cmap=pyplot.cm.Blues)
    # # pyplot.savefig(settings['EXPERIMENT_NAME'] + '.png')
    # pyplot.show()

    # # rewrite data structure
    # feature_selection_result_dict = {}
    # test_train_indices_list = []
    # for feature_subset, test_indices, train_indices in feature_subsets:
    #     test_train_indices_list.append((test_indices, train_indices))
    #     for selection_method, selected_features in feature_subset.items():
    #         selected_feature_subset_list = feature_selection_result_dict.get(
    #             selection_method, []
    #         )
    #         selected_feature_subset_list.append(selected_features)
    #         feature_selection_result_dict[
    #             selection_method
    #         ] = selected_feature_subset_list
    # assert len(test_train_indices_list) == settings.N_FOLDS_OUTER_CV

    matthews_list = []
    balanced_accuracy_score_list = []
    number_of_features = []
    # for hyper_k in [3, 5, 7, 9, 11]:
    #     min_difference = np.asarray(range(1, 70, 1)) / 100
    #     for d in min_difference:
    #         print(d)

    # matthews_list = []
    # balanced_accuracy_score_list = []
    # number_of_features = []
    # for hyper_k in [3, 5, 7, 9, 11]:
    #     min_difference = np.asarray(range(1, 70, 1)) / 100
    #     for d in min_difference:
    #         print(d)
    #         for (
    #             feature_selection_method,
    #             selected_feature_subsets,
    #         ) in feature_selection_result_dict.items():
    #             if feature_selection_method == "reverse_lasso":
    #                 union, intersect = evaluate_selected_features(selected_feature_subsets)
    #                 print("##############################", feature_selection_method, ": ")
    #                 print("union_of_features: ")
    #                 print(len(union), union)
    #                 # evaluate_proportion_of_selected_features(union)
    #                 print("intersection_of_features: ")
    #                 print(len(intersect), intersect)
    #                 # evaluate_proportion_of_selected_features(intersect)
    #
    #                 temp_selected_feature_subsets = []
    #                 if feature_selection_method == "reverse_lasso":
    #                     for feature_subset in selected_feature_subsets:
    #                         subset = {}
    #                         for feature in feature_subset.keys():
    #                             if (
    #                                 feature_subset[feature][1]
    #                                 >= feature_subset[feature][0] + d
    #                             ):
    #                                 subset[feature] = (
    #                                     feature_subset[feature][1]
    #                                     - feature_subset[feature][0]
    #                                 )
    #                         temp_selected_feature_subsets.append(subset)
    #                     selected_feature_subsets = temp_selected_feature_subsets
    #
    #                     union2, intersect2 = evaluate_selected_features(
    #                         selected_feature_subsets
    #                     )
    #                     print(d, "union:", len(union2), union2)
    #                     print(d, "intersection:", len(intersect2), intersect2)
    #                     number_of_features.append(len(intersect2))
    #
    #                 intersect_dict_list = []
    #                 # Iterate over all the items in dictionary and filter items which has even keys
    #                 for k in range(len(test_train_indices_list)):
    #                     intersect_dict = dict()
    #                     for (key, value) in selected_feature_subsets[k].items():
    #                         # Check if key is even then add pair to new dictionary
    #                         if key in intersect:
    #                             intersect_dict[key] = value
    #                     intersect_dict_list.append(intersect_dict)
    #
    #                 # outer cross-validation to validate selected feature subsets
    #                 # loo = LeaveOneOut()
    #                 results = Parallel(n_jobs=1, verbose=5)(
    #                     delayed(weighted_knn.validate)(
    #                         clustered_data_df.iloc[test_indices, :],
    #                         clustered_data_df.iloc[train_indices, :],
    #                         # intersect,
    #                         intersect_dict_list[outer_cv_loop_iteration],
    #                         hyper_k,
    #                     )
    #                     for outer_cv_loop_iteration, (
    #                         test_indices,
    #                         train_indices,
    #                     ) in enumerate(test_train_indices_list)
    #                 )
    #                 assert len(results) == len(test_train_indices_list)
    #
    #                 # flatten lists of results from k-fold cross-validation
    #                 predicted_classes = []
    #                 true_classes = []
    #                 for predicted_classes_sublist, true_classes_sublist in results:
    #                     predicted_classes.extend(predicted_classes_sublist)
    #                     true_classes.extend(true_classes_sublist)
    #                 # predicted_classes = [
    #                 #     item[0] for sublist in results for item in sublist
    #                 # ]
    #                 assert len(predicted_classes) == data_df.shape[0]
    #                 if feature_selection_method == "reverse_lasso":
    #                     matthews_list.append(
    #                         matthews_corrcoef(true_classes, predicted_classes)
    #                     )
    #                     balanced_accuracy_score_list.append(
    #                         balanced_accuracy_score(true_classes, predicted_classes)
    #                     )
    #                 # print(predicted_classes)
    #                 # print(true_classes)
    #                 print(
    #                     "matthews",
    #                     matthews_corrcoef(true_classes, predicted_classes),
    #                     "accuracy",
    #                     accuracy_score(true_classes, predicted_classes),
    #                     "f1_score",
    #                     f1_score(true_classes, predicted_classes),
    #                     "balanced_accuracy_score",
    #                     balanced_accuracy_score(true_classes, predicted_classes),
    #                 )
    # print("matthews:")
    # print(max(matthews_list))
    # print(matthews_list)
    # print(list(zip(matthews_list, number_of_features)))
    # print("balanced_accuracy_score:")
    # print(max(balanced_accuracy_score_list))
    # print(balanced_accuracy_score_list)
    # print(list(zip(balanced_accuracy_score_list, number_of_features)))
