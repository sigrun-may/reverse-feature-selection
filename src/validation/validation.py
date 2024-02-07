from typing import Dict, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    brier_score_loss,
    top_k_accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier

from src.validation.stability_estimator import get_stability
from weighted_manhattan_distance import WeightedManhattanDistance

# data_name = "overlapping_500_3"
data_name = "colon_loo"

list_of_methods = ["standard", "shap", "reverse"]
activate_p_value = False
p_value_threshold = 0.05

activate_threshold = True
threshold = 0.1


def evaluate_feature_selection(
<<<<<<< HEAD
        feature_selection_result_dict,
=======
    feature_selection_result_dict,
>>>>>>> main
):
    metrics_per_method_dict = {}

    cv_result_list = feature_selection_result_dict["method_result_dict"]["rf_cv_result"]

    # iterate over feature selection algorithms
    for method, cv_result_list in feature_selection_result_dict["method_result_dict"].items():
        if method != "rf_cv_result":
            continue
        print(method)
        performance_metrics_dict = {}
        importance_matrix_reverse = np.empty((len(cv_result_list), cv_result_list[0].shape[0]))
        importance_matrix_standard = np.empty_like(importance_matrix_reverse)
        importance_matrix_shap = np.empty_like(importance_matrix_standard)

        importance_matrix_reverse_t = np.empty((len(cv_result_list), cv_result_list[0].shape[0]))
        importance_matrix_standard_t = np.empty_like(importance_matrix_reverse)
        importance_matrix_shap_t = np.empty_like(importance_matrix_standard)

        # iterate over cross-validation results
        for i, cv_iteration_result_pd in enumerate(cv_result_list):
            _normalize_feature_subsets(cv_iteration_result_pd)
            importance_matrix_reverse[i, :] = cv_iteration_result_pd["reverse_ts"].values
            importance_matrix_standard[i, :] = cv_iteration_result_pd["standard_ts"].values
            importance_matrix_shap[i, :] = cv_iteration_result_pd["shap_ts"].values

            importance_matrix_reverse_t[i, :] = cv_iteration_result_pd["reverse_t"].values
            importance_matrix_standard_t[i, :] = cv_iteration_result_pd["standard_t"].values
            importance_matrix_shap_t[i, :] = cv_iteration_result_pd["shap_t"].values

        # monitor stability of selection
        binary_importance_matrix_standard = np.zeros_like(importance_matrix_standard)
        binary_importance_matrix_reverse = np.zeros_like(importance_matrix_reverse)
        binary_importance_matrix_shap = np.zeros_like(importance_matrix_shap)
        assert binary_importance_matrix_reverse.shape == binary_importance_matrix_standard.shape

        binary_importance_matrix_standard[importance_matrix_standard.nonzero()] = 1
        binary_importance_matrix_reverse[importance_matrix_reverse.nonzero()] = 1
        binary_importance_matrix_shap[importance_matrix_shap.nonzero()] = 1
        performance_metrics_dict["reverse_stability"] = get_stability(binary_importance_matrix_reverse)
        performance_metrics_dict["standard_stability"] = get_stability(binary_importance_matrix_standard)
        performance_metrics_dict["shap_stability"] = get_stability(binary_importance_matrix_shap)

        binary_importance_matrix_standard_t = np.zeros_like(importance_matrix_standard_t)
        binary_importance_matrix_reverse_t = np.zeros_like(importance_matrix_reverse_t)
        binary_importance_matrix_shap_t = np.zeros_like(importance_matrix_shap_t)
        assert binary_importance_matrix_reverse_t.shape == binary_importance_matrix_standard_t.shape

        binary_importance_matrix_standard_t[importance_matrix_standard_t.nonzero()] = 1
        binary_importance_matrix_reverse_t[importance_matrix_reverse_t.nonzero()] = 1
        binary_importance_matrix_shap_t[importance_matrix_shap_t.nonzero()] = 1
        performance_metrics_dict["reverse_stability_t"] = get_stability(binary_importance_matrix_reverse_t)
        performance_metrics_dict["standard_stability_t"] = get_stability(binary_importance_matrix_standard_t)
        performance_metrics_dict["shap_stability_t"] = get_stability(binary_importance_matrix_shap_t)

        reverse_subset = np.sum(importance_matrix_reverse, axis=0)
        standard_subset = np.sum(importance_matrix_standard, axis=0)
        assert reverse_subset.shape == standard_subset.shape
        df = pd.DataFrame(columns=["reverse"], data=reverse_subset)
        df["standard"] = standard_subset
        trimmed_standard_subset = np.argwhere(standard_subset > 0.1)
        trimmed_reverse_subset = np.argwhere(reverse_subset > 0.1)
        num_selected_reverse = np.count_nonzero(reverse_subset)
        num_selected_standard = np.count_nonzero(standard_subset)
        robust_selected_reverse = np.sum(binary_importance_matrix_reverse, axis=0)
        robust_selected_standard = np.sum(binary_importance_matrix_standard, axis=0)
        num_robust_selected_reverse = len(
            robust_selected_reverse[robust_selected_reverse == binary_importance_matrix_reverse.shape[0]]
        )
        num_robust_selected_standard = len(
            robust_selected_standard[robust_selected_standard == binary_importance_matrix_standard.shape[0]]
        )
        num_close_to_robust_selected_reverse = len(
            robust_selected_reverse[robust_selected_reverse >= binary_importance_matrix_reverse.shape[0] - 1]
        )
        num_close_to_robust_selected_standard = len(
            robust_selected_standard[robust_selected_standard >= binary_importance_matrix_standard.shape[0] - 1]
        )

    return metrics_per_method_dict


def evaluate_cross_validation_results(cv_result_list: list) -> dict:
<<<<<<< HEAD
    """ Evaluate cross-validation results.
=======
    """Evaluate cross-validation results.
>>>>>>> main
    Args:
        cv_result_list: List containing pandas Dataframes with corresponding cross-validation results.
    Returns:
        List containing pandas Dataframes with calculated cross-validation results.
    """
    # calculate feature weights for reverse feature selection
    cv_result_list = _calculate_reverse_feature_selection_results(cv_result_list)
    # normalize feature importance results
    _normalize_results(cv_result_list)

    feature_names = cv_result_list[0].index.values
    result_dict = {}
    selected_feature_subsets_df = pd.DataFrame()
    for method in list_of_methods:
        importance_matrix = _get_importance_matrix(cv_result_list, method)
        binary_importance_matrix = _binarize_feature_importance_matrix(importance_matrix)

        # calculate stability of feature selection
        result_dict[f"stability_{method}"] = get_stability(binary_importance_matrix)
        # calculate number of robust features
        result_dict[f"robust_{method}"] = np.count_nonzero(
            np.int_(binary_importance_matrix.sum(axis=0)) == binary_importance_matrix.shape[0]
        )
        result_dict[f"robust_{method}_names"] = str(
            feature_names[np.where(np.int_(binary_importance_matrix.sum(axis=0)) == binary_importance_matrix.shape[0])]
        )
        selected_feature_subsets_df[method] = importance_matrix.sum(axis=0)
        # calculate number of selected features
        result_dict[f"selected_{method}"] = np.count_nonzero(importance_matrix.sum(axis=0))
    result_dict["selected_feature_subsets"] = selected_feature_subsets_df
    return result_dict


def _calculate_reverse_feature_selection_results(cv_result_list: list):
<<<<<<< HEAD
    """ Calculate feature importance results for reverse feature selection.
=======
    """Calculate feature importance results for reverse feature selection.
>>>>>>> main
    Args:
        cv_result_list: List containing pandas Dataframes with corresponding cross-validation results.

    Returns:
        List containing pandas Dataframes with calculated cross-validation results.
    """
    list_of_calculated_reverse_feature_selection_results = []
    for cv_iteration_result_df in cv_result_list:
        reverse_feature_importance = _calculate_feature_weights(cv_iteration_result_df)
        list_of_calculated_reverse_feature_selection_results.append(reverse_feature_importance)
    return list_of_calculated_reverse_feature_selection_results


def _normalize_results(cv_result_list: list):
<<<<<<< HEAD
    """ Normalize feature importance results.
=======
    """Normalize feature importance results.
>>>>>>> main
    Args:
        cv_result_list: List containing pandas Dataframes with corresponding cross-validation results.

    Returns:
        normalized cv_result_list: List containing pandas Dataframes with corresponding normalized cross-validation results.
    """
    for cv_iteration_result_df in cv_result_list:
        _normalize_feature_subsets(cv_iteration_result_df)
    return


<<<<<<< HEAD
def _calculate_stability_of_feature_selection(
        cv_result_list: list, method: str
)-> float:
    """ Calculate stability of feature selection.
=======
def _calculate_stability_of_feature_selection(cv_result_list: list, method: str) -> float:
    """Calculate stability of feature selection.
>>>>>>> main
    Args:
        cv_result_list: List containing pandas Dataframes with corresponding cross-validation results.
        method: Method to evaluate. Either "standard", "shap" or "reverse".

    Returns:
        stability: Stability of feature selection.
    """

    importance_matrix = _get_importance_matrix(cv_result_list, method)
    binary_importance_matrix = _binarize_feature_importance_matrix(importance_matrix)
    stability = get_stability(binary_importance_matrix)
    return stability


def _binarize_feature_importance_matrix(_importance_matrix: np.ndarray) -> np.ndarray:
    """Binarize feature importance matrix.

    Args:
        _importance_matrix: Matrix containing the feature importance of each feature for each cross-validation iteration.

    Returns:
        binary_importance_matrix: Matrix containing the binarized feature importance of each feature for each cross-validation iteration.

    """
    binary_importance_matrix = np.zeros_like(_importance_matrix)
    binary_importance_matrix[_importance_matrix.nonzero()] = 1
    return binary_importance_matrix


def _get_importance_matrix(_cv_result_list: list, key: str) -> np.ndarray:
    """Extract feature importance matrix from cross-validation results.

    Args:
        _cv_result_list: List containing pandas Dataframes with corresponding cross-validation results.
        key: Key of the column to retrieve the importance from in each Dataframe.

    Returns:
        importance_matrix: Matrix containing the feature importance of each feature for each cross-validation iteration.



    """
    importance_matrix = np.empty((len(_cv_result_list), _cv_result_list[0].shape[0]))

    # iterate over cross-validation results
    for i, cv_iteration_result_pd in enumerate(_cv_result_list):
        _normalize_feature_subsets(cv_iteration_result_pd)
        importance_matrix[i, :] = cv_iteration_result_pd[key].values

    return importance_matrix


def _normalize_feature_subsets(feature_selection_result_pd):
    # normalize standard feature selection result
    assert np.min(feature_selection_result_pd["standard"].values) >= 0
    standard_feature_importance = feature_selection_result_pd["standard"].values / max(
        feature_selection_result_pd["standard"].values
    )
    shap_feature_importance = feature_selection_result_pd["shap_values"].values / max(
        feature_selection_result_pd["shap_values"].values
    )

    feature_selection_result_pd["standard_ts"] = standard_feature_importance
    feature_selection_result_pd["shap_ts"] = shap_feature_importance

    # normalize reverse feature selection result
    reverse_feature_importance = _calculate_feature_weights(feature_selection_result_pd)
    feature_selection_result_pd["reverse_ts"] = reverse_feature_importance

    feature_selection_result_pd["standard_t"] = apply_threshold(standard_feature_importance, threshold)
    feature_selection_result_pd["shap_t"] = apply_threshold(shap_feature_importance, threshold)

    # normalize reverse feature selection result
    feature_selection_result_pd["reverse_t"] = apply_threshold(reverse_feature_importance, threshold)


def apply_threshold(feature_importance_array: np.ndarray, _threshold: float) -> np.ndarray:
    """Drop feature importance below threshold.

    Args:
        _threshold:
        feature_importance_array:

    Returns:

    """
    if np.sum(feature_importance_array) > 0:
        for fi_index in range(feature_importance_array.size):
            # cut values below _threshold
            if feature_importance_array[fi_index] < _threshold:
                feature_importance_array[fi_index] = 0.0
    assert not 0 < np.min(feature_importance_array) < threshold
    assert np.sum(feature_importance_array) > 0
    return feature_importance_array


def _calculate_feature_weights(result_pd):
    metrics_labeled_training_np = result_pd["labeled"].values
    metrics_unlabeled_training_np = result_pd["unlabeled"].values
    p_values_tt = result_pd["p_values_tt"].values

    # TODO ensure maximized metric or implement also case minimizing
    feature_weights_np = abs(metrics_labeled_training_np - metrics_unlabeled_training_np) / abs(
        metrics_unlabeled_training_np
    )
    for i in range(metrics_unlabeled_training_np.size):
        if metrics_unlabeled_training_np[i] == np.inf:
            feature_weights_np[i] = 0.0
        else:
            # TODO percentage difference as feature weight?
            feature_weights_np[i] = abs(metrics_labeled_training_np[i] - metrics_unlabeled_training_np[i]) / abs(
                metrics_unlabeled_training_np[i]
            )
        if activate_p_value and p_values_tt[i] > p_value_threshold:
            feature_weights_np[i] = 0.0
    assert np.min(feature_weights_np) >= 0.0

    normalized_feature_weights_np = feature_weights_np / np.max(feature_weights_np)
    assert np.min(normalized_feature_weights_np) >= 0.0
    assert np.max(normalized_feature_weights_np) == 1.0
    return normalized_feature_weights_np


def calculate_performance(test_train_sets, feature_names, importance_matrix, meta_data) -> Dict:
    """

    Args:
        test_train_sets:
        feature_names:
        importance_matrix:
        meta_data:

    Returns:

    """
    if not np.sum(importance_matrix) > 0:
        # no features were selected
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
    binary_importance_matrix = np.zeros_like(importance_matrix)
    binary_importance_matrix[importance_matrix.nonzero()] = 1
    performance_metrics_dict["stability"] = get_stability(binary_importance_matrix)

    performance_metrics_dict["robust features"] = np.count_nonzero(
        np.int_(binary_importance_matrix.sum(axis=0)) == binary_importance_matrix.shape[0]
    )
    performance_metrics_dict["robust feature names"] = str(
        feature_names[np.where(np.int_(binary_importance_matrix.sum(axis=0)) == binary_importance_matrix.shape[0])]
    )

    # calculate correctness for artificial data
    if "artificial" in meta_data["data"]["input_data_path"]:
        _measure_correctness_of_feature_selection(
            performance_metrics_dict,
            binary_importance_matrix,
            feature_names[feature_importance.nonzero()],
            meta_data,
        )
    return performance_metrics_dict


def classify_feature_subsets(
<<<<<<< HEAD
        test_train_sets,
        # TODO series?
        selected_feature_names,
        weights,
        meta_data,
=======
    test_train_sets,
    # TODO series?
    selected_feature_names,
    weights,
    meta_data,
>>>>>>> main
):
    classified_classes = []
    true_classes = []
    macro_auc = []
    macro_logloss = []
    macro_brier_score_loss = []
    macro_top_k_accuracy_score = []
    for test_df, train_df in test_train_sets:
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
            top_k_accuracy_score(
                list(test_df["label"]),
                class_probabilities[:, meta_data["data"]["pos_label"]],
            )
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
        "micro_matthews": matthews_corrcoef(true_classes, predicted_classes),
        "micro_accuracy": accuracy_score(true_classes, predicted_classes),
        "micro_f1_score": f1_score(
            true_classes,
            predicted_classes,
            pos_label=meta_data_dict["data"]["pos_label"],
        ),
        "micro_balanced_accuracy_score": balanced_accuracy_score(true_classes, predicted_classes),
    }


def _measure_correctness_of_feature_selection(
<<<<<<< HEAD
        performance_metrics_dict,
        selected_features_matrix,
        selected_feature_names,
        meta_data,
=======
    performance_metrics_dict,
    selected_features_matrix,
    selected_feature_names,
    meta_data,
>>>>>>> main
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


def _trim_and_scale_robust_features(_importance_matrix, _threshold, reverse=False, close_to_robust=False):
    """Drop feature importance below threshold.
    Args:
        _threshold:
        _importance_matrix:

    Returns:

    """
    assert np.sum(_importance_matrix) > 0
    binary_importance_matrix = np.zeros_like(_importance_matrix)
    binary_importance_matrix[_importance_matrix.nonzero()] = 1

    if close_to_robust:
        index_of_robust_features = np.where(
            np.int_(binary_importance_matrix.sum(axis=0)) >= binary_importance_matrix.shape[0] - 1
        )
        if not len(index_of_robust_features) > 0:
            index_of_robust_features = np.where(
                np.int_(binary_importance_matrix.sum(axis=0)) == binary_importance_matrix.shape[0]
            )
    else:
        index_of_robust_features = np.where(
            np.int_(binary_importance_matrix.sum(axis=0)) == binary_importance_matrix.shape[0]
        )

    robust_features_matrix = np.zeros_like(_importance_matrix)
    robust_features_matrix[:, index_of_robust_features] = _importance_matrix[:, index_of_robust_features]
    assert np.array_equal(
        robust_features_matrix[:, index_of_robust_features],
        _importance_matrix[:, index_of_robust_features],
    )
    if reverse:
        return _trim_and_scale_feature_weight_matrix(robust_features_matrix, _threshold)
    else:
        return _trim_and_scale_feature_importance_matrix(robust_features_matrix, _threshold)


def _trim_and_scale_feature_importance_matrix(_importance_matrix, _threshold):
    """Drop feature importance below threshold.
    Args:
        _threshold:
        _importance_matrix:

    Returns:

    """
    processed_matrix = np.zeros_like(_importance_matrix)
    if np.sum(_importance_matrix) > 0:
        for row in range(_importance_matrix.shape[0]):
            # normalize with softmax similar function
            normalized_fi = _normalize_feature_importance(_importance_matrix[row, :])
            # cut values below _threshold
            normalized_fi[normalized_fi < _threshold] = 0.0
            if sum(normalized_fi[normalized_fi.nonzero()]) > 0:
                assert np.min(normalized_fi[normalized_fi.nonzero()]) >= _threshold
            processed_matrix[row, :] = normalized_fi
    return processed_matrix


def _normalize_feature_importance(_feature_importance):
    # # normalize with softmax function
    _feature_importance = _feature_importance * 1000

    if np.isclose(np.sum(_feature_importance), 0):
        normalized_fi = _feature_importance
    else:
        # # ensure the sum to be more than one
        # assert np.sum(_feature_importance) > 1
        normalized_fi = softmax(_feature_importance)
        assert np.isclose(np.sum(normalized_fi), 1), np.sum(normalized_fi)
    return normalized_fi


def _trim_and_scale_feature_weight_matrix(_importance_matrix, _threshold):
    """Drop feature importance below threshold.
    Args:
        _threshold:
        _importance_matrix:

    Returns:

    """
    processed_matrix = np.zeros_like(_importance_matrix)
    if np.sum(_importance_matrix) > 0:
        for row in range(_importance_matrix.shape[0]):
            # cut values below _threshold
            fi = _importance_matrix[row, :]
            fi[fi < _threshold] = 0.0
            if sum(fi) > 0:
                assert np.min(fi[fi.nonzero()]) >= _threshold
                # normalize with softmax similar function
                processed_matrix[row, :] = _normalize_feature_importance(fi)
    return processed_matrix


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
            if "reverse" not in selection_method:
                if selection_method not in shap_values_dict:
                    shap_values_dict[selection_method] = selected_features["shap_values"]
                else:
                    shap_values_dict[selection_method] = np.concatenate(
                        (
                            shap_values_dict[selection_method],
                            selected_features["shap_values"],
                        )
                    )
                if selection_method not in macro_fi_dict:
                    macro_fi_dict[selection_method] = selected_features["macro_feature_importances"]
                else:
                    macro_fi_dict[selection_method] = np.concatenate(
                        (
                            macro_fi_dict[selection_method],
                            selected_features["macro_feature_importances"],
                        )
                    )
                if selected_features["micro_feature_importance"] is not None:
                    if selection_method not in micro_fi_dict:
                        micro_fi_dict[selection_method] = selected_features["micro_feature_importance"]
                    else:
                        micro_fi_dict[selection_method] = np.concatenate(
                            (
                                micro_fi_dict[selection_method],
                                selected_features["micro_feature_importance"],
                            )
                        )
            else:
                # save reverse selection methods
                if selection_method not in reverse_subsets_dict:
                    # calculate metric distances for reverse selection
                    reverse_subsets_dict[selection_method] = _get_selected_features_array(selected_features)
                else:
                    reverse_subsets_dict[selection_method] = np.vstack(
                        (
                            reverse_subsets_dict[selection_method],
                            _get_selected_features_array(selected_features),
                        )
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
<<<<<<< HEAD
                    micro_fi_dict[selection_method].shape
                    == (
                            meta_data_dict["cv"]["n_outer_folds"],
                            len(meta_data_dict["data"]["columns"]) - 1,
                    )
=======
                micro_fi_dict[selection_method].shape
                == (
                    meta_data_dict["cv"]["n_outer_folds"],
                    len(meta_data_dict["data"]["columns"]) - 1,
                )
>>>>>>> main
            ):
                micro_fi_dict.pop(selection_method)
            else:
                assert micro_fi_dict[selection_method].shape == (
                    meta_data_dict["cv"]["n_outer_folds"],
                    len(meta_data_dict["data"]["columns"]) - 1,  # remove label
                ), micro_fi_dict[selection_method].shape

    for selection_method in reverse_subsets_dict:
        assert reverse_subsets_dict[selection_method].shape == (
            meta_data_dict["cv"]["n_outer_folds"],
            len(meta_data_dict["data"]["columns"]) - 1,
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
            selection.append(r2 * 10 * (r2 - r2_unlabeled))
        else:
            selection.append(0.0)
    return np.array(selection)


def test_trim_and_scale_robust_features():
    only_ones = np.ones((10, 20))
    stability = get_stability(only_ones)
    assert stability == 1, stability

    perfect_stability = np.concatenate((np.ones((10, 8)), np.zeros((10, 120))), axis=1)
    assert perfect_stability.shape == (10, 128)
    perfect_stability_metric = get_stability(perfect_stability)
    assert perfect_stability_metric == 1, perfect_stability_metric

    perfect_stability2 = np.concatenate((np.ones((10, 1)), np.zeros((10, 120))), axis=1)
    assert perfect_stability2.shape == (10, 121)
    perfect_stability_metric2 = get_stability(perfect_stability2)
    assert perfect_stability_metric2 == 1, perfect_stability_metric2

    # print(get_stability(perfect_stability))
    not_stable = get_stability(np.zeros((10, 20)))
    assert not_stable == 0, not_stable


cv_result_dict = joblib.load(f"../../results/{data_name}_result_dict.pkl")
evaluated_results_dict = evaluate_cross_validation_results(cv_result_dict["method_result_dict"]["rf_cv_result"])
print(evaluated_results_dict["stability_reverse"])
