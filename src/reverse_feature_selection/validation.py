from typing import List, Dict
from weighted_manhattan_distance import WeightedManhattanDistance
from sklearn.neighbors import KNeighborsClassifier, VALID_METRICS
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import MinMaxScaler
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    brier_score_loss,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from stability_estimator import get_stability
import utils


def measure_correctness_of_feature_selection(
    performance_metrics_dict, selected_features_matrix, selected_feature_names, meta_data
):
    bm = 0
    pseudo = 0
    random = 0
    for feature_name in selected_feature_names:
        if "bm" in feature_name:
            bm += 1
        elif "pseudo" in feature_name:
            pseudo += 1
        elif "random" in feature_name:
            random += 1
    performance_metrics_dict["number_of_relevant_features"] = bm
    performance_metrics_dict["number_of_pseudo_features"] = pseudo
    performance_metrics_dict["number_of_random_features"] = random

    relevant_features = map(lambda x: "bm" in x, list(meta_data["data"]["columns"]))
    performance_metrics_dict["relevant_features"] = sum(relevant_features)
    performance_metrics_dict["selected_features"] = len(selected_feature_names)

    feature_names = meta_data["data"]["columns"][1:]
    true_values = []
    for feature in feature_names:
        if "bm" in feature:
            true_values.append(1)
        else:
            true_values.append(0)

    robustness_vector = selected_features_matrix.sum(axis=0)
    assert len(robustness_vector) == len(feature_names)
    selected_features = np.zeros_like(robustness_vector)
    selected_features[robustness_vector.nonzero()] = 1
    f1 = f1_score(true_values, selected_features)
    print("f1:", f1)
    performance_metrics_dict["correctness f1"] = f1
    return performance_metrics_dict


def _scale_feature_importance_matrix(_importance_matrix, _threshold):
    """Drop feature importance below threshold.
    Args:
        _threshold:
        _importance_matrix:

    Returns:

    """
    if np.sum(_importance_matrix) > 0:
        for row in range(_importance_matrix.shape[0]):
            # weighted_scaled_fi = (_importance_matrix[row, :] * robustness)/np.max(_importance_matrix[row, :] * robustness)
            weighted_scaled_fi = _importance_matrix[row, :]/np.max(_importance_matrix[row, :])
            assert math.isclose(np.max(weighted_scaled_fi), 1.0), weighted_scaled_fi
            # weighted_scaled_fi = scaled_fi * robustness
            weighted_scaled_fi[weighted_scaled_fi < _threshold] = 0.0
            assert np.min(weighted_scaled_fi[weighted_scaled_fi.nonzero()]) >= _threshold
            weighted_scaled_fi = weighted_scaled_fi * 1000
            normalized_fi = weighted_scaled_fi / np.sum(weighted_scaled_fi)
            assert np.isclose(np.sum(normalized_fi), 1), np.sum(normalized_fi)
            _importance_matrix[row, :] = normalized_fi
    return _importance_matrix


def _trim_and_scale_feature_importance_matrix(_importance_matrix, _binary_importance_matrix, _threshold):
    """Drop feature importance below threshold.
    Args:
        _binary_importance_matrix:
        _threshold:
        _importance_matrix:

    Returns:

    """
    # binary_importance_matrix = np.zeros_like(raw_importance_matrix)
    # binary_importance_matrix[_importance_matrix.nonzero()] = 1


    # best_robustness_matrix = np.zeros_like(_importance_matrix)
    # best_method = None
    # for method, binary_importance_matrix in feature_selection_result['raw_robustness'].items():
    #     # binary_importance_matrix = np.zeros_like(raw_importance_matrix)
    #     # binary_importance_matrix[_importance_matrix.nonzero()] = 1
    #     if np.sum(binary_importance_matrix) > np.sum(best_robustness_matrix):
    #         best_robustness_matrix = binary_importance_matrix
    #         best_method = method
    # print('best_method', best_method)
    # print(np.sum(best_robustness_matrix).nonzero())

    assert np.sum(_importance_matrix) > 0
    # binary_importance_matrix = np.zeros_like(_importance_matrix)
    # binary_importance_matrix[_importance_matrix.nonzero()] = 1
    # robustness = binary_importance_matrix.sum(axis=0)

    # robust_feature_matrix = np.zeros_like(_importance_matrix)
    # index_of_robust_features = np.where(
    #     np.int_(best_robustness_matrix.sum(axis=0))
    #     == best_robustness_matrix.shape[0]
    # )
    robust_feature_matrix = np.zeros_like(_importance_matrix)
    index_of_robust_features = np.where(
        np.int_(_binary_importance_matrix.sum(axis=0))
        == _binary_importance_matrix.shape[0]
    )
    robust_feature_matrix[:, index_of_robust_features] = _importance_matrix[:, index_of_robust_features]
    assert np.array_equal(robust_feature_matrix[:, index_of_robust_features],_importance_matrix[:, index_of_robust_features])
    # _importance_matrix = robust_feature_matrix
    # if np.sum(_importance_matrix) > 0:
    #     for row in range(_importance_matrix.shape[0]):
    #         # weighted_scaled_fi = (_importance_matrix[row, :] * robustness)/np.max(_importance_matrix[row, :] * robustness)
    #         weighted_scaled_fi = _importance_matrix[row, :]/np.max(_importance_matrix[row, :])
    #         assert math.isclose(np.max(weighted_scaled_fi), 1.0), weighted_scaled_fi
    #         # weighted_scaled_fi = scaled_fi * robustness
    #         weighted_scaled_fi[weighted_scaled_fi < _threshold] = 0.0
    #         assert np.min(weighted_scaled_fi[weighted_scaled_fi.nonzero()]) >= _threshold
    #         weighted_scaled_fi = weighted_scaled_fi * 1000
    #         normalized_fi = weighted_scaled_fi / np.sum(weighted_scaled_fi)
    #         assert np.isclose(np.sum(normalized_fi), 1), np.sum(normalized_fi)
    #         _importance_matrix[row, :] = normalized_fi
    # return _importance_matrix
    return _scale_feature_importance_matrix(robust_feature_matrix, _threshold)


def scale_feature_importances(_importance_matrix):
    """Scale feature importances.
    Args:
        _importance_matrix:

    Returns:

    """
    if not np.sum(_importance_matrix) > 0:
        return np.mean(_importance_matrix, axis=0)
    else:
        feature_importances = np.mean(_importance_matrix, axis=0)
        feature_importances = feature_importances/np.sum(feature_importances)
        assert np.isclose(np.sum(feature_importances), 1), np.sum(feature_importances)
        return feature_importances


def evaluate_feature_selection(
    feature_selection_result,
    meta_data,
):
    (
        feature_selection_result_dict,
        test_train_sets,
    ) = _extract_test_data_and_results(feature_selection_result, meta_data)

    metrics_per_method_dict = {}
    unlabeled_feature_names = np.asarray(meta_data["data"]["columns"][1:])
    selected_features_df = pd.DataFrame(index=unlabeled_feature_names)

    # select reverse and standard feature selection
    for feature_selection_algorithm, result_dict in feature_selection_result_dict.items():
        # select applied embedded feature selection method
        for embedded_feature_selection_method, importance_matrix in result_dict.items():
            key = f"{feature_selection_algorithm}_{embedded_feature_selection_method}"
            print(
                "##############################",
                feature_selection_algorithm,
                "",
                embedded_feature_selection_method,
                ": ",
            )
            assert isinstance(importance_matrix, np.ndarray)
            assert np.min(importance_matrix) >= 0.0
            # feature_selection_dict["raw_importances_matrix"] = importance_matrix
            # feature_selection_dict["binary_importances_matrix"] = np.zeros_like(importance_matrix)
            # feature_selection_dict["binary_importances_matrix"][importance_matrix.nonzero()] = 1
            # feature_selection_dict["robustness"] = feature_selection_dict["binary_importances_matrix"].sum(axis=0)
            # used_features_matrix = np.zeros_like(importance_matrix)
            # used_features_matrix[importance_matrix.nonzero()] = 1

            # # drop feature importance below validation threshold
            # if "reverse" not in feature_selection_algorithm:
            #     for row in range(importance_matrix.shape[0]):
            #         normalized_fi = importance_matrix[row, :] / np.max(importance_matrix[row, :])
            #         feature_selection_dict["normalized_importances"] = normalized_fi
            #         normalized_fi[normalized_fi < meta_data["validation"]["standard_threshold"]] = 0.0
            #         importance_matrix[row, :] = normalized_fi
            #     assert (
            #         np.min(importance_matrix[importance_matrix > 0]) >= meta_data["validation"]["standard_threshold"]
            #     )

            # drop feature importance below validation threshold
            # feature_selection_dict["trimmed_importances_matrix"] = trim_and_scale_feature_importances(
            #     feature_selection_dict, meta_data["validation"]["standard_threshold"])

            # adapt feature importances

            # weighting robustness
            # TODO quadrierte robustness/ sinus... / log
            # if "robustness" in feature_selection_algorithm:
            #     print(
            #         "robustness: ",
            #         importance_matrix.sum(axis=0),
            #         " from ",
            #         importance_matrix.shape[0],
            #     )
            #     feature_selection_dict["binary_raw_selection"] = []

            # cumulated_feature_importance = np.sum(np.array(importance_matrix), axis=0)

            # feature_selection_dict["mean_feature_importance"] = np.mean(
            #     np.array(feature_selection_dict["raw_importances_matrix"]), axis=0)
            # feature_selection_dict["mean_trimmed_feature_importance"] = np.mean(
            #     np.array(feature_selection_dict["trimmed_importances_matrix"]), axis=0)
            # assert len(unlabeled_feature_names) == feature_selection_dict["mean_feature_importance"].size == \
            #        feature_selection_dict["mean_trimmed_feature_importance"].size
            #
            # selected_features_df[f"{key}_raw"] = feature_selection_dict["mean_feature_importance"] * \
            #                                      feature_selection_dict["robustness"]
            # selected_features_df[f"{key}_trimmed"] = feature_selection_dict["mean_trimmed_feature_importance"] * \
            #                                          feature_selection_dict["robustness"]
            #
            # # # print important features
            # # selected_feature_names = unlabeled_feature_names[cumulated_feature_importance.nonzero()]
            # # nonzero_feature_importances = cumulated_feature_importance[cumulated_feature_importance.nonzero()]
            # #
            # # scaled_feature_importances = nonzero_feature_importances / np.sum(nonzero_feature_importances)
            # #
            # # assert np.isclose(np.sum(scaled_feature_importances), 1), np.sum(scaled_feature_importances)
            # # assert type(scaled_feature_importances) == np.ndarray, type(scaled_feature_importances)
            # # assert len(selected_feature_names) == cumulated_feature_importance.nonzero()[0].size
            # # named_importances = list(zip(selected_feature_names, scaled_feature_importances))
            # # sorted_importances = utils.sort_list_of_tuples_by_index(named_importances, ascending=False)
            # # feature_selection_dict["selected_features"] = sorted_importances
            # # print("number of selected features:", len(sorted_importances))
            # # print("importances:", sorted_importances)
            # # # TODO save to csv
            #
            # # calculate performance evaluation metrics
            # performance_metrics_dict = classify_feature_subsets(
            #     test_train_sets,
            #     selected_feature_names=unlabeled_feature_names[
            #         feature_selection_dict["mean_trimmed_feature_importance"].nonzero()],
            #     scaled_feature_importances=feature_selection_dict["mean_trimmed_feature_importance"],
            #     meta_data=meta_data,
            # )
            # # monitor stability of selection
            # performance_metrics_dict["raw_stability"] = get_stability(feature_selection_dict["raw_importances_matrix"])
            # performance_metrics_dict["trimmed_stability"] = get_stability(
            #     feature_selection_dict["trimmed_importances_matrix"])
            # performance_metrics_dict["robust features"] = np.count_nonzero(
            #     np.int_(feature_selection_dict["raw_importances"].sum(axis=0))
            #     == feature_selection_dict["raw_importances"].shape[0]
            # )
            # index_of_robust_features = np.where(
            #     np.int_(feature_selection_dict["raw_importances"].sum(axis=0))
            #     == feature_selection_dict["raw_importances"].shape[0]
            # )
            # print('robust features: ', unlabeled_feature_names[index_of_robust_features])
            #
            # # calculate correctness
            # measure_correctness_of_feature_selection(
            #     performance_metrics_dict, used_features_matrix, selected_feature_names, meta_data
            # )

            # metrics_per_method_dict[f"{key}_raw"] = calculate_performance(
            #     test_train_sets, unlabeled_feature_names, importance_matrix, meta_data
            # )
            binary_importance_matrix = np.zeros_like(importance_matrix)
            binary_importance_matrix[importance_matrix.nonzero()] = 1
            # metrics_per_method_dict[f"{key}_binary"] = calculate_performance(
            #     test_train_sets, unlabeled_feature_names, binary_importance_matrix, meta_data
            # )
            trimmed_importance_matrix = _trim_and_scale_feature_importance_matrix(
                importance_matrix, binary_importance_matrix, meta_data["validation"]["standard_threshold"]
            )
            metrics_per_method_dict[f"{key}_trimmed"] = calculate_performance(
                test_train_sets, unlabeled_feature_names, trimmed_importance_matrix, meta_data
            )
            raw_importance_matrix = _scale_feature_importance_matrix(importance_matrix, 0.0)
            metrics_per_method_dict[f"{key}_raw"] = calculate_performance(
                test_train_sets, unlabeled_feature_names, raw_importance_matrix, meta_data
            )
            scaled_importance_matrix = _scale_feature_importance_matrix(importance_matrix, meta_data["validation"]["standard_threshold"])
            metrics_per_method_dict[f"{key}_raw"] = calculate_performance(
                test_train_sets, unlabeled_feature_names, scaled_importance_matrix, meta_data
            )

            # trimmed_binary_importance_matrix = np.zeros_like(trimmed_importance_matrix)
            # trimmed_binary_importance_matrix[trimmed_importance_matrix.nonzero()] = 1
            # metrics_per_method_dict[f"{key}_trimmed_binary"] = calculate_performance(
            #     test_train_sets, unlabeled_feature_names, trimmed_importance_matrix, meta_data
            # )

            selected_features_df[f"{key}_raw"] = scale_feature_importances(importance_matrix)
            selected_features_df[f"{key}_trimmed"] = scale_feature_importances(trimmed_importance_matrix)
            # selected_features_df[f"{key}_raw_robustness"] = binary_importance_matrix.sum(axis=0)
            # selected_features_df[f"{key}_trimmed_robustness"] = trimmed_binary_importance_matrix.sum(axis=0)

            # predicted_classes, true_classes = classify_feature_subsets(
            #     test_train_sets,
            #     selected_feature_names,
            #     scaled_feature_importances,
            #     meta_data,
            # )
            #
            # metrics_per_method_dict[key] = calculate_micro_metrics(
            #     predicted_classes, true_classes
            # )

    selected_features_df.sort_values(by=selected_features_df.columns[0], ascending=False, inplace=True)
    selected_features_df[selected_features_df.values.sum(axis=1) != 0].to_csv(meta_data["path_selected_features"])
    return metrics_per_method_dict


def calculate_performance(test_train_sets, feature_names, importance_matrix, meta_data):
    if not np.sum(importance_matrix) > 0:
        return {}

    assert np.min(importance_matrix) >= 0.0
    assert len(feature_names) == importance_matrix.shape[1]
    feature_importance = np.mean(np.array(importance_matrix), axis=0)

    # calculate performance evaluation metrics
    performance_metrics_dict = classify_feature_subsets(
        test_train_sets,
        feature_names[feature_importance.nonzero()],
        feature_importance[feature_importance.nonzero()],
        meta_data,
    )
    # monitor stability of selection
    binary_importances_matrix = np.zeros_like(importance_matrix)
    binary_importances_matrix[importance_matrix.nonzero()] = 1
    performance_metrics_dict["stability"] = get_stability(binary_importances_matrix)

    performance_metrics_dict["robust features"] = np.count_nonzero(
        np.int_(binary_importances_matrix.sum(axis=0)) == binary_importances_matrix.shape[0]
    )
    print(
        "robust features: ",
        feature_names[np.where(np.int_(binary_importances_matrix.sum(axis=0)) == binary_importances_matrix.shape[0])],
    )

    # calculate correctness
    measure_correctness_of_feature_selection(
        performance_metrics_dict, binary_importances_matrix, feature_names[feature_importance.nonzero()], meta_data
    )
    return performance_metrics_dict


def classify_feature_subsets(
    test_train_sets,
    selected_feature_names,
    weights,
    meta_data,
):
    classified_classes = []
    true_classes = []
    macro_auc = []
    macro_logloss = []
    macro_brier_score_loss = []
    macro_top_k_accuracy_score = []
    for test_df, train_df in test_train_sets:
        # train_df = pd.DataFrame(data=train, columns=meta_data["data"]["columns"])
        # test_df = pd.DataFrame(data=test, columns=meta_data["data"]["columns"])

        train_data = train_df[selected_feature_names]
        test_data = test_df[selected_feature_names]

        knn_clf = KNeighborsClassifier(
            n_neighbors=meta_data["validation"]["k_neighbors"],
            weights=meta_data["validation"]["knn_method"],
            # metric='manhattan',
            metric=WeightedManhattanDistance(weights=weights),
            algorithm="brute",
        )
        knn_clf.fit(train_data, train_df["label"])

        class_probabilities = knn_clf.predict_proba(test_data)
        macro_auc.append(roc_auc_score(list(test_df["label"]), class_probabilities[:, 1]))
        macro_logloss.append(log_loss(list(test_df["label"]), class_probabilities))
        macro_brier_score_loss.append(
            brier_score_loss(
                list(test_df["label"]),
                class_probabilities[:, meta_data["data"]["pos_label"]],
                pos_label=meta_data["data"]["pos_label"],
            )
        )
        macro_top_k_accuracy_score.append(
            top_k_accuracy_score(list(test_df["label"]), class_probabilities[:, meta_data["data"]["pos_label"]])
        )

        classified_classes.extend(knn_clf.predict(test_data))
        true_classes.extend(list(test_df["label"]))

    assert len(classified_classes) == len(true_classes)
    metrics_dict = calculate_micro_metrics(classified_classes, true_classes, meta_data)
    metrics_dict["macro_auc"] = np.mean(macro_auc)
    metrics_dict["macro_logloss"] = np.mean(macro_logloss)
    metrics_dict["macro_brier_score_loss"] = np.mean(macro_brier_score_loss)
    metrics_dict["macro_top_k_accuracy_score"] = np.mean(macro_top_k_accuracy_score)
    return metrics_dict


def _extract_test_data_and_results(feature_selection_result, meta_data_dict):
    preprocessed_test_train_sets = []
    reverse_subsets_dict = {}
    shap_values_dict = {}
    macro_fi_dict = {}
    micro_fi_dict = {}

    for feature_selection_dict, test_data, train_data in feature_selection_result:
        preprocessed_test_train_sets.append((test_data, train_data))

        # rewrite data structure
        for selection_method, selected_features in feature_selection_dict.items():
            if not selected_features:
                continue

            # save standard methods
            if selection_method not in shap_values_dict:
                shap_values_dict[selection_method] = selected_features["shap_values"]
            else:
                shap_values_dict[selection_method] = np.concatenate(
                    (shap_values_dict[selection_method], selected_features["shap_values"])
                )
            if selection_method not in macro_fi_dict:
                macro_fi_dict[selection_method] = selected_features["macro_feature_importances"]
            else:
                macro_fi_dict[selection_method] = np.concatenate(
                    (macro_fi_dict[selection_method], selected_features["macro_feature_importances"])
                )
            if selected_features["micro_feature_importance"] is not None:
                if selection_method not in micro_fi_dict:
                    micro_fi_dict[selection_method] = selected_features["micro_feature_importance"]
                else:
                    micro_fi_dict[selection_method] = np.concatenate(
                        (micro_fi_dict[selection_method], selected_features["micro_feature_importance"])
                    )
            # save reverse selection methods
            if selection_method not in reverse_subsets_dict:
                # calculate metric distances for reverse selection
                reverse_subsets_dict[selection_method] = _get_selected_features_array(selected_features)
            else:
                reverse_subsets_dict[selection_method] = np.vstack(
                    (reverse_subsets_dict[selection_method], _get_selected_features_array(selected_features))
                )

    # asserts and cleanup
    for selection_method in macro_fi_dict:
        assert (
            macro_fi_dict[selection_method].shape
            == shap_values_dict[selection_method].shape
            == (meta_data_dict["cv"]["n_outer_folds"] * meta_data_dict["cv"]["n_inner_folds"]),
            len(meta_data_dict["data"]["columns"]) - 1,  # remove label
        ), (
            len(meta_data_dict["data"]["columns"]) - 1,
            meta_data_dict["cv"]["n_outer_folds"] * meta_data_dict["cv"]["n_inner_folds"],
        )
        if micro_fi_dict and (selection_method in micro_fi_dict):
            # remove method with incomplete matrix
            if not (
                micro_fi_dict[selection_method].shape
                == (
                    meta_data_dict["cv"]["n_outer_folds"],
                    len(meta_data_dict["data"]["columns"]) - 1,
                )
            ):
                micro_fi_dict.pop(selection_method)
            else:
                assert micro_fi_dict[selection_method].shape == (
                    meta_data_dict["cv"]["n_outer_folds"],
                    len(meta_data_dict["data"]["columns"]) - 1,  # remove label
                ), micro_fi_dict[selection_method].shape

    for selection_method in reverse_subsets_dict:
        assert (
            reverse_subsets_dict[selection_method].shape
            == (
                meta_data_dict["cv"]["n_outer_folds"],
                len(meta_data_dict["data"]["columns"]) - 1,
            )
        )

    result_dict = {
        "reverse_subsets": reverse_subsets_dict,
        "shap_values": shap_values_dict,
        "macro_fi": macro_fi_dict,
        "micro_fi": micro_fi_dict,
    }
    return result_dict, preprocessed_test_train_sets


def _get_selected_features_array(selected_feature_subset):
    selection = []
    for feature, (r2_unlabeled, r2) in selected_feature_subset.items():
        # TODO minimize and maximize unterscheiden
        assert r2 >= 0.0
        if r2 > r2_unlabeled:
            selection.append(r2 - r2_unlabeled)
        else:
            selection.append(0.0)
    return np.array(selection)


def _calculate_weights_reverse_feature_selection(selected_feature_subset, threshold):
    weighted_selected_feature_subset = []
    robustness = []
    # try:
    #     for feature, (r2_unlabeled, r2, _) in selected_feature_subset.items():
    #
    #         # for feature, (r2_unlabeled, r2) in feature_subset.items():
    #         # TODO minimize and maximize unterscheiden
    #         # difference = r2 - r2_unlabeled
    #         # if difference > delta:
    #         #     subset[feature] = difference
    #         assert r2 >= 0.0
    #         if r2 > r2_unlabeled:
    #             robustness.append(1)
    #             if r2_unlabeled < 0:
    #                 r2_unlabeled = 0
    #             d = r2 - r2_unlabeled
    #             d *= r2
    #             if d > threshold:
    #                 weighted_selected_feature_subset.append(d)
    #             else:
    #                 weighted_selected_feature_subset.append(0)
    #         else:
    #             robustness.append(0)
    #             weighted_selected_feature_subset.append(0)
    #             # weighted_selected_feature_subset[feature] = (r2 - r2_unlabeled,
    #             # None)
    # except:
    for feature, (r2_unlabeled, r2) in selected_feature_subset.items():
        # TODO minimize and maximize unterscheiden
        assert r2 >= 0.0
        if r2 > r2_unlabeled:
            robustness.append(1)
            # if r2_unlabeled < 0:
            #     r2_unlabeled = 0
            d = r2 - r2_unlabeled
            d *= r2
            if d > threshold:
                weighted_selected_feature_subset.append(d)
            else:
                weighted_selected_feature_subset.append(0)
        else:
            robustness.append(0)
            weighted_selected_feature_subset.append(0)

    return np.array(weighted_selected_feature_subset), np.array(robustness)


def calculate_micro_metrics(predicted_classes, true_classes, meta_data_dict):
    print(
        "matthews",
        matthews_corrcoef(true_classes, predicted_classes),
        "accuracy",
        accuracy_score(true_classes, predicted_classes),
        "f1_score",
        f1_score(
            true_classes,
            predicted_classes,
            pos_label=meta_data_dict["data"]["pos_label"],
        ),
        "balanced_accuracy_score",
        balanced_accuracy_score(true_classes, predicted_classes),
        print(classification_report(true_classes, predicted_classes)),
        # print(precision_recall_fscore_support(true_classes, predicted_classes)),
    )
    return {
        "micro_matthews": matthews_corrcoef(true_classes, predicted_classes),
        "micro_accuracy": accuracy_score(true_classes, predicted_classes),
        "micro_f1_score": f1_score(true_classes, predicted_classes, pos_label=meta_data_dict["data"]["pos_label"]),
        "micro_balanced_accuracy_score": balanced_accuracy_score(true_classes, predicted_classes),
    }
