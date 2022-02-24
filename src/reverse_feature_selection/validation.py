from typing import List, Dict

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

from src.reverse_feature_selection import weighted_knn
from src.reverse_feature_selection.misc import evaluate_selected_features


def _extract_indices_and_results(feature_selection_result):
    # rewrite data structure
    result_dict = {}
    test_train_indices = []
    for feature_selection_dict, test_indices, train_indices in feature_selection_result:
        test_train_indices.append((test_indices, train_indices))
        for selection_method, selected_features in feature_selection_dict.items():
            selected_feature_subset_list = result_dict.get(selection_method, [])
            selected_feature_subset_list.append(selected_features)
            result_dict[selection_method] = selected_feature_subset_list
    # assert len(test_train_indices) == settings.N_FOLDS_OUTER_CV
    return result_dict, test_train_indices


def _calculate_weights_for_reverse_lasso(
    selected_feature_subsets, delta=None, factor=None
):
    weighted_selected_feature_subsets = []

    # set identity element
    if factor is None:
        factor = 1
    if delta is None:
        delta = 0

    for feature_subset in selected_feature_subsets:
        subset = {}
        for feature, (r2_unlabeled, r2) in feature_subset.items():
            # TODO minimize and maximize unterscheiden
            # difference = r2 - r2_unlabeled
            # if difference > delta:
            #     subset[feature] = difference

            if (r2 > (r2_unlabeled + delta)) and (r2 > (r2_unlabeled * factor)):
                subset[feature] = (r2 - r2_unlabeled, None)
        weighted_selected_feature_subsets.append(subset)
    return weighted_selected_feature_subsets


def _select_intersect_from_feature_subsets(selected_feature_subsets, intersect):
    intersect_dict_list = []
    for subset_dict in selected_feature_subsets:
        intersect_dict = dict()
        for key, value in subset_dict.items():
            if key in intersect:
                intersect_dict[key] = value
        intersect_dict_list.append(intersect_dict)
    return intersect_dict_list


def classify_feature_subsets(
    k_neighbors, test_train_indices_list, feature_subset_dict_list, data_df
):
    assert len(feature_subset_dict_list) == len(test_train_indices_list)
    assert isinstance(feature_subset_dict_list, list)
    assert isinstance(test_train_indices_list, list)
    results = []
    for outer_cv_loop_iteration, (
        test_indices,
        train_indices,
    ) in enumerate(test_train_indices_list):
        assert len(test_indices) + len(train_indices) == data_df.shape[0]
        results.append(
            weighted_knn.validate_standard(
                data_df.iloc[test_indices, :],
                data_df.iloc[train_indices, :],
                # intersect,
                feature_subset_dict_list[outer_cv_loop_iteration],
                k_neighbors,
            )
        )
    assert len(results) == len(test_train_indices_list)

    # flatten lists of results from k-fold cross-validation
    predicted_classes = []
    true_classes = []
    logloss_list = []
    auc_list = []
    for predicted_classes_sublist, true_classes_sublist, logloss, auc in results:
        predicted_classes.extend(predicted_classes_sublist)
        true_classes.extend(true_classes_sublist)
        logloss_list.append(logloss)
        auc_list.append(auc)

    print("mean logloss: ", np.mean(logloss_list))
    print("mean auc: ", np.mean(auc_list))

    return predicted_classes, true_classes


def calculate_metrics(predicted_classes, true_classes):
    print(
        "matthews",
        matthews_corrcoef(true_classes, predicted_classes),
        "accuracy",
        accuracy_score(true_classes, predicted_classes),
        "f1_score",
        f1_score(true_classes, predicted_classes, pos_label=0),
        "balanced_accuracy_score",
        balanced_accuracy_score(true_classes, predicted_classes),
    )
    return {
        "matthews": matthews_corrcoef(true_classes, predicted_classes),
        "accuracy": accuracy_score(true_classes, predicted_classes),
        "f1_score": f1_score(true_classes, predicted_classes),
        "balanced_accuracy_score": balanced_accuracy_score(
            true_classes, predicted_classes
        ),
    }


def evaluate_feature_selection_methods(
    feature_selection_result_dict,
    test_train_indices_list,
    data_df,
    k_neighbors,
):
    metrics_per_method_dict = {}
    for (
        feature_selection_method,
        selected_feature_subsets,
    ) in feature_selection_result_dict.items():
        print("##############################", feature_selection_method, ": ")
        # if feature_selection_method == "standard_lasso":
        #     continue

        if feature_selection_method == "reverse_lasso":
            selected_feature_subsets = _calculate_weights_for_reverse_lasso(
                selected_feature_subsets, delta=0.0, factor=1
            )
        # calculate performance evaluation metrics
        predicted_classes, true_classes = classify_feature_subsets(
            k_neighbors, test_train_indices_list, selected_feature_subsets, data_df
        )
        metrics_per_method_dict[feature_selection_method] = calculate_metrics(
            predicted_classes, true_classes
        )

        # evaluate robustness
        evaluate_selected_features(selected_feature_subsets)

        # rank selected features
        # print(neue_methode(selected_feature_subsets))
        # feature_importance_robustness_dict = neue_methode(selected_feature_subsets)
        # feature_importance_list = [
        #     (k, v) for k, v in feature_importance_robustness_dict.items()
        # ]
        #
        # # getting length of list of tuples
        # lst = len(feature_importance_list)
        # for i in range(0, lst):
        #     for j in range(0, lst - i - 1):
        #         if feature_importance_list[j][1] < feature_importance_list[j + 1][1]:
        #             temp = feature_importance_list[j]
        #             feature_importance_list[j] = feature_importance_list[j + 1]
        #             feature_importance_list[j + 1] = temp
        # print(feature_importance_list)

    return metrics_per_method_dict


def neue_methode(all_feature_subsets: List[Dict[str, float]]):
    d = {}
    for feature_subset in all_feature_subsets:
        for k, v in feature_subset.items():
            if k not in d:
                d[k] = (0.0, 0)
            d[k] = (d[k][0] + v[0], d[k][1] + 1)
    return d


def evaluate_selected_subsets(
    data_df, selected_feature_subsets, test_train_indices_list, k_neighbors
):
    # union, intersect = evaluate_selected_features(selected_feature_subsets)

    # if intersect and settings.CLASSIFY_INTERSECT_ONLY:
    #     print('intersect for classification: ')
    #     selected_feature_subsets = _select_intersect_from_feature_subsets(
    #         selected_feature_subsets, intersect
    #     )
    # else:
    #     print('union for classification: ')
    #     selected_feature_subsets = _select_intersect_from_feature_subsets(
    #         selected_feature_subsets, union
    #     )

    predicted_classes, true_classes = classify_feature_subsets(
        k_neighbors, test_train_indices_list, selected_feature_subsets, data_df
    )
    return calculate_metrics(predicted_classes, true_classes)


def evaluate_feature_selection_method():
    return
