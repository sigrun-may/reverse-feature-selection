from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report
)
from stability_estimator import get_stability
from src.reverse_feature_selection import weighted_knn
import utils

# from src.reverse_feature_selection.misc import evaluate_selected_features


def _extract_indices_and_results(feature_selection_result, meta_data_dict):
    # rewrite data structure
    test_train_sets = []

    reverse_subsets_dict = {}
    shap_values_dict = {}
    macro_fi_dict = {}
    micro_fi_dict = {}

    for feature_selection_dict, test_data, train_data in feature_selection_result:
        test_train_sets.append((test_data, train_data))
        for selection_method, selected_features in feature_selection_dict.items():
            if "reverse" not in selection_method:
                if selection_method not in shap_values_dict:
                    shap_values_dict[selection_method] = selected_features[
                        "shap_values"
                    ]
                else:
                    shap_values_dict[selection_method] = np.concatenate(
                        (
                            shap_values_dict[selection_method],
                            selected_features["shap_values"],
                        )
                    )

                if selection_method not in macro_fi_dict:
                    macro_fi_dict[selection_method] = selected_features[
                        "macro_feature_importances"
                    ]
                else:
                    macro_fi_dict[selection_method] = np.concatenate(
                        (
                            macro_fi_dict[selection_method],
                            selected_features["macro_feature_importances"],
                        )
                    )

                if selection_method not in micro_fi_dict:
                    micro_fi_dict[selection_method] = selected_features[
                        "micro_feature_importance"
                    ]
                elif selected_features["micro_feature_importance"] is not None:
                    micro_fi_dict[selection_method] = np.vstack(
                        (
                            micro_fi_dict[selection_method],
                            selected_features["micro_feature_importance"],
                        )
                    )
                else:
                    micro_fi_dict[selection_method] = None
            else:
                weighted_features = _calculate_weights_reverse_feature_selection(
                    selected_features, delta = 0.1
                )
                assert (
                    len(weighted_features) == len(meta_data_dict["data"]["columns"]) - 1
                )
                if selection_method not in reverse_subsets_dict:
                    reverse_subsets_dict[selection_method] = weighted_features
                else:
                    reverse_subsets_dict[selection_method] = np.vstack(
                        (
                            reverse_subsets_dict[selection_method],
                            weighted_features,
                        )
                    )

    for selection_method in macro_fi_dict:
        assert (
            macro_fi_dict[selection_method].shape
            == shap_values_dict[selection_method].shape
            == (
                (
                    meta_data_dict["cv"]["n_outer_folds"]
                    * meta_data_dict["cv"]["n_inner_folds"]
                ),
                len(meta_data_dict["data"]["columns"]) - 1,  # remove label
            )
        ), (
            len(meta_data_dict["data"]["columns"]) - 1,
            meta_data_dict["cv"]["n_outer_folds"]
            * meta_data_dict["cv"]["n_inner_folds"],
        )
        if micro_fi_dict[selection_method] is not None:
            assert micro_fi_dict[selection_method].shape == (
                meta_data_dict["cv"]["n_outer_folds"],
                len(meta_data_dict["data"]["columns"]) - 1,  # remove label
            ), micro_fi_dict[selection_method].shape
        else:
            micro_fi_dict.pop(selection_method)

    for selection_method in reverse_subsets_dict:
        assert reverse_subsets_dict[selection_method].shape == (
            meta_data_dict["cv"]["n_outer_folds"],
            len(meta_data_dict["data"]["columns"]) - 1,
        )

    rd = {
        "reverse_subsets": reverse_subsets_dict,
        "shap_values": shap_values_dict,
        "macro_fi": macro_fi_dict,
        "micro_fi": micro_fi_dict,
    }
    return (
        rd,
        test_train_sets
    )


def _extract_indices_and_results_reverse(feature_selection_result):
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


# def _calculate_weights_reverse_feature_selection(
#     selected_feature_subsets, delta=None, factor=None
# ):
#     weighted_selected_feature_subsets = []
#
#     # set identity element
#     if factor is None:
#         factor = 1
#     if delta is None:
#         delta = 0
#
#     for feature_subset in selected_feature_subsets:
#         subset = {}
#         for feature, (r2_unlabeled, r2) in feature_subset.items():
#             # TODO minimize and maximize unterscheiden
#             # difference = r2 - r2_unlabeled
#             # if difference > delta:
#             #     subset[feature] = difference
#
#             if (r2 > (r2_unlabeled + delta)) and (r2 > (r2_unlabeled * factor)):
#                 subset[feature] = (r2 - r2_unlabeled, None)
#         weighted_selected_feature_subsets.append(subset)
#     return weighted_selected_feature_subsets


def _calculate_weights_reverse_feature_selection(
    selected_feature_subset, delta=None, factor=None
):
    weighted_selected_feature_subsets = []

    # set identity element
    if factor is None:
        factor = 1
    if delta is None:
        delta = 0

    weighted_selected_feature_subset = []
    for feature, (r2_unlabeled, r2) in selected_feature_subset.items():

        # for feature, (r2_unlabeled, r2) in feature_subset.items():
        # TODO minimize and maximize unterscheiden
        # difference = r2 - r2_unlabeled
        # if difference > delta:
        #     subset[feature] = difference

        if (r2 > (r2_unlabeled + delta)) and (r2 > (r2_unlabeled * factor)):
            if r2_unlabeled < 0:
                r2_unlabeled = 0
            weighted_selected_feature_subset.append(r2 - r2_unlabeled)
        else:
            weighted_selected_feature_subset.append(0)
            # weighted_selected_feature_subset[feature] = (r2 - r2_unlabeled,
            # None)
    return np.array(weighted_selected_feature_subset)


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
    k_neighbors, test_train_sets, selected_features, weights, meta_data
):

    labeled_selected_features = ['label']
    labeled_selected_features.extend(selected_features)

    results = []
    for outer_cv_loop_iteration, (
        test,
        train,
    ) in enumerate(test_train_sets):
        test_df = pd.DataFrame(test, columns = meta_data['data']['columns'])
        train_df = pd.DataFrame(train, columns = meta_data['data']['columns'])

        results.append(
            weighted_knn.validate_standard(
                test_df[labeled_selected_features],
                train_df[labeled_selected_features],
                k_neighbors,
                weights,
            )
        )
    assert len(results) == len(test_train_sets)

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
    print(classification_report(true_classes, predicted_classes))

    return predicted_classes, true_classes


def _classify_feature_subsets(
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


def evaluate_feature_selection(
    feature_selection_result,
    data_df,
    meta_data,
):
    (
        feature_selection_result_dict,
        test_train_sets,
    ) = _extract_indices_and_results(feature_selection_result, meta_data)

    metrics_per_method_dict = {}
    for feature_selection_method, result_dict in feature_selection_result_dict.items():
        for feature_selection_algorithm, selection_result in result_dict.items():
            print("##############################",
                  feature_selection_method, " ",
                  feature_selection_algorithm, ": ")
            assert np.ndarray == type(selection_result)
            # monitor robustness of selection
            used_features_matrix = np.zeros_like(selection_result)
            used_features_matrix[selection_result.nonzero()] = 1

            print(
                "stab:",
                get_stability(used_features_matrix),
            )

            # calculate importances
            feature_idx = np.sum(np.array(selection_result), axis=0)
            unlabeled_feature_names = np.asarray(meta_data['data'][
                                                     'columns'][1:])
            assert len(unlabeled_feature_names) == feature_idx.size
            selected_features = unlabeled_feature_names[feature_idx.nonzero()]
            nonzero_features = feature_idx[feature_idx.nonzero()]
            scaled_feature_importances = nonzero_features / np.sum(
                nonzero_features)
            assert np.isclose(np.sum(scaled_feature_importances), 1), np.sum(scaled_feature_importances)
            assert type(scaled_feature_importances) == np.ndarray, type(scaled_feature_importances)
            assert len(selected_features) == feature_idx.nonzero()[0].size
            named_importances = list(zip(selected_features, nonzero_features))
            sorted_importances = utils.sort_list_of_tuples_by_index(
                named_importances, ascending = False)
            print('number of selected features:', len(sorted_importances))
            print('importances:', sorted_importances)

            # calculate performance evaluation metrics
            predicted_classes, true_classes = classify_feature_subsets(
                meta_data["validation"]["k_neighbors"],
                test_train_sets,
                selected_features,
                nonzero_features,
                # scaled_feature_importances,
                meta_data
            )
            key = f"{feature_selection_method}_{feature_selection_algorithm}"
            metrics_per_method_dict[key] = calculate_metrics(
                predicted_classes, true_classes
            )




    # for (
    #     feature_selection_method,
    #     selected_feature_subsets,
    # ) in feature_selection_result_dict.items():
    #     print("##############################", feature_selection_method, ": ")
    #
    #     assert (
    #         len(robustness_dict[feature_selection_method])
    #         == len(meta_data["data"]["columns"]) - 1  # remove label count
    #     )
    #
    #     if "reverse" in feature_selection_method:
    #         selected_feature_subsets = _calculate_weights_reverse_feature_selection(
    #             selected_feature_subsets,
    #             delta=meta_data["validation"]["delta"],
    #             factor=meta_data["validation"]["factor"],
    #         )
    #         subset_vector = [len(subset) for subset in selected_feature_subsets]
    #         assert len(subset_vector) == meta_data["cv"]["n_outer_folds"]  # lambda
    #
    #     else:
    #         assert (
    #             binary_selections_dict[feature_selection_method].shape[1]
    #             == len(meta_data["data"]["columns"]) - 1
    #         )
    #         subset_vector = np.sum(
    #             binary_selections_dict[feature_selection_method], axis=1
    #         )
    #         assert subset_vector.ndim == 1
    #         assert (
    #             len(subset_vector)
    #             == meta_data["cv"]["n_outer_folds"] * meta_data["cv"]["n_inner_folds"]
    #         )  # lambda
    #     print(
    #         "stab:",
    #         get_stability(
    #             robustness_dict[feature_selection_method],
    #             subset_vector,
    #         ),
    #     )



        # evaluate robustness
        # evaluate_selected_features(selected_feature_subsets)

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
