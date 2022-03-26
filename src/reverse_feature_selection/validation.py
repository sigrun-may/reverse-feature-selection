from typing import List, Dict
from weighted_manhattan_distance import WeightedManhattanDistance
from sklearn.neighbors import KNeighborsClassifier, VALID_METRICS
from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

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
)
from stability_estimator import get_stability
import utils


def measure_correctness_of_feature_selection(selected_features_matrix, meta_data):
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
    return f1


def evaluate_feature_selection(
    feature_selection_result,
    meta_data,
):
    (
        feature_selection_result_dict,
        test_train_sets,
    ) = _extract_test_data_and_results(feature_selection_result, meta_data)

    metrics_per_method_dict = {}
    selected_features_dict = {}
    unlabeled_feature_names = np.asarray(meta_data["data"]["columns"][1:])
    selected_features_df = pd.DataFrame(index=unlabeled_feature_names)

    for feature_selection_method, result_dict in feature_selection_result_dict.items():
        for feature_selection_algorithm, importance_matrix in result_dict.items():
            key = f"{feature_selection_method}_{feature_selection_algorithm}"
            feature_selection_dict = {}
            print(
                "##############################",
                feature_selection_method,
                "",
                feature_selection_algorithm,
                ": ",
            )
            assert isinstance(importance_matrix, np.ndarray)
            feature_selection_dict["raw_importances"] = importance_matrix

            # drop feature importance below validation threshold
            if "reverse" not in feature_selection_method:
                fi_threshold = meta_data["validation"]["threshold"]
                for row in range(importance_matrix.shape[0]):
                    normalized_fi = importance_matrix[row, :] / np.max(
                        importance_matrix[row, :]
                    )
                    feature_selection_dict["normalized_importances"] = normalized_fi
                    normalized_fi[normalized_fi < fi_threshold] = 0.0
                    importance_matrix[row, :] = normalized_fi
                assert np.min(importance_matrix[importance_matrix > 0]) >= fi_threshold

            feature_selection_dict["trimmed_importances"] = importance_matrix

            # monitor stability of selection
            used_features_matrix = np.zeros_like(importance_matrix)
            used_features_matrix[importance_matrix.nonzero()] = 1
            stability = get_stability(used_features_matrix)
            print("stability:", stability)
            feature_selection_dict["binary_trimmed_selection"] = used_features_matrix

            # adapt feature importances

            # weighting robustness
            # TODO quadrierte robustness/ sinus... / log
            if "robustness" in feature_selection_method:
                print(
                    "robustness: ",
                    importance_matrix.sum(axis=0),
                    " from ",
                    importance_matrix.shape[0],
                )
                feature_selection_dict["binary_raw_selection"] = []

            # cumulated_feature_importance = np.sum(np.array(importance_matrix), axis=0)
            assert np.min(importance_matrix) >= 0.0
            mean_feature_importance = np.mean(np.array(importance_matrix), axis=0)
            # robustness_vector = used_features_matrix.sum(axis=0)
            # print(robustness_vector[robustness_vector.nonzero()])
            raw_robustness = (
                feature_selection_dict["raw_importances"].sum(axis=0)
                / importance_matrix.shape[0]
            )
            print(raw_robustness[raw_robustness.nonzero()])
            cumulated_feature_importance = mean_feature_importance * raw_robustness
            assert len(unlabeled_feature_names) == cumulated_feature_importance.size
            selected_features_df[key] = cumulated_feature_importance
            feature_selection_dict[
                "weighted_feature_importance"
            ] = cumulated_feature_importance

            # print important features
            selected_feature_names = unlabeled_feature_names[
                cumulated_feature_importance.nonzero()
            ]
            nonzero_feature_importances = cumulated_feature_importance[
                cumulated_feature_importance.nonzero()
            ]

            scaled_feature_importances = nonzero_feature_importances / np.sum(
                nonzero_feature_importances
            )

            assert np.isclose(np.sum(scaled_feature_importances), 1), np.sum(
                scaled_feature_importances
            )
            assert type(scaled_feature_importances) == np.ndarray, type(
                scaled_feature_importances
            )
            assert (
                len(selected_feature_names)
                == cumulated_feature_importance.nonzero()[0].size
            )
            named_importances = list(
                zip(selected_feature_names, scaled_feature_importances)
            )
            sorted_importances = utils.sort_list_of_tuples_by_index(
                named_importances, ascending=False
            )
            feature_selection_dict["selected_features"] = sorted_importances
            print("number of selected features:", len(sorted_importances))
            print("importances:", sorted_importances)
            # TODO save to csv

            # calculate performance evaluation metrics
            performance_metrics_dict = classify_feature_subsets(
                test_train_sets,
                selected_feature_names,
                scaled_feature_importances,
                meta_data,
            )
            performance_metrics_dict["stability"] = stability

            # calculate correctness
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

            relevant_features = map(
                lambda x: "bm" in x, list(meta_data["data"]["columns"])
            )
            performance_metrics_dict["relevant_features"] = sum(relevant_features)
            performance_metrics_dict["selected_features"] = len(selected_feature_names)
            performance_metrics_dict[
                "correctness f1"
            ] = measure_correctness_of_feature_selection(
                used_features_matrix, meta_data
            )

            metrics_per_method_dict[key] = performance_metrics_dict

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

    selected_features_df.sort_values(
        by=selected_features_df.columns[0], ascending=False, inplace=True
    )
    selected_features_df[selected_features_df.values.sum(axis=1) != 0].to_csv(
        meta_data["path_selected_features"]
    )
    return metrics_per_method_dict


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
    for test, train in test_train_sets:
        train_df = pd.DataFrame(data=train, columns=meta_data["data"]["columns"])
        test_df = pd.DataFrame(data=test, columns=meta_data["data"]["columns"])

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
        macro_auc.append(
            roc_auc_score(list(test_df["label"]), class_probabilities[:, 1])
        )
        macro_logloss.append(log_loss(list(test_df["label"]), class_probabilities))

        classified_classes.extend(knn_clf.predict(test_data))
        true_classes.extend(list(test_df["label"]))

    assert len(classified_classes) == len(true_classes)
    metrics_dict = calculate_micro_metrics(classified_classes, true_classes, meta_data)
    metrics_dict["macro_auc"] = np.mean(macro_auc)
    metrics_dict["macro_logloss"] = np.mean(macro_logloss)
    return metrics_dict


def _extract_test_data_and_results(feature_selection_result, meta_data_dict):
    test_train_sets = []
    reverse_subsets_dict = {}
    raw_robustness = {}
    shap_values_dict = {}
    macro_fi_dict = {}
    micro_fi_dict = {}

    for feature_selection_dict, test_data, train_data in feature_selection_result:
        test_train_sets.append((test_data, train_data))

        # rewrite data structure
        for selection_method, selected_features in feature_selection_dict.items():
            if not selected_features:
                continue

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

                if selected_features["micro_feature_importance"] is not None:
                    if selection_method not in micro_fi_dict:
                        micro_fi_dict[selection_method] = selected_features[
                            "micro_feature_importance"
                        ]
                    else:
                        micro_fi_dict[selection_method] = np.vstack(
                            (
                                micro_fi_dict[selection_method],
                                selected_features["micro_feature_importance"],
                            )
                        )
            else:
                (
                    weighted_features,
                    reverse_robustness,
                ) = _calculate_weights_reverse_feature_selection(
                    selected_features,
                    threshold=meta_data_dict["validation"]["threshold"],
                )
                assert (
                    len(weighted_features)
                    == len(reverse_robustness)
                    == len(meta_data_dict["data"]["columns"]) - 1
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

                if selection_method not in raw_robustness:
                    raw_robustness[selection_method] = reverse_robustness
                else:
                    raw_robustness[selection_method] = np.vstack(
                        (
                            raw_robustness[selection_method],
                            reverse_robustness,
                        )
                    )

    # asserts
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
            == raw_robustness[selection_method].shape
            == (
                meta_data_dict["cv"]["n_outer_folds"],
                len(meta_data_dict["data"]["columns"]) - 1,
            )
        )

    result_dict = {
        "reverse_subsets": reverse_subsets_dict,
        "raw_robustness": raw_robustness,
        "shap_values": shap_values_dict,
        "macro_fi": macro_fi_dict,
        "micro_fi": micro_fi_dict,
    }
    return result_dict, test_train_sets


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

        # for feature, (r2_unlabeled, r2) in feature_subset.items():
        # TODO minimize and maximize unterscheiden
        # difference = r2 - r2_unlabeled
        # if difference > delta:
        #     subset[feature] = difference
        assert r2 >= 0.0
        if r2 > r2_unlabeled:
            robustness.append(1)
            if r2_unlabeled < 0:
                r2_unlabeled = 0
            d = r2 - r2_unlabeled
            d *= r2
            if d > threshold:
                weighted_selected_feature_subset.append(d)
            else:
                weighted_selected_feature_subset.append(0)
        else:
            robustness.append(0)
            weighted_selected_feature_subset.append(0)
            # weighted_selected_feature_subset[feature] = (r2 - r2_unlabeled,
            # None)
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
    )
    return {
        "matthews": matthews_corrcoef(true_classes, predicted_classes),
        "accuracy": accuracy_score(true_classes, predicted_classes),
        "f1_score": f1_score(true_classes, predicted_classes),
        "balanced_accuracy_score": balanced_accuracy_score(
            true_classes, predicted_classes
        ),
    }
