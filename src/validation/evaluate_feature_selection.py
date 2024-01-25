import math
import os
import pickle
from typing import Literal, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

import stability_estimator


def evaluate(pickled_result_path: str, feature_selection_methods: list[str], thresholds: dict = None) -> dict:
    """
    Evaluates the feature selection results.

    This function loads the pickled results from the specified path, extracts the feature selection results,
    and evaluates them based on the provided feature selection methods and thresholds.

    Args:
        pickled_result_path: The path to the pickled results file.
        feature_selection_methods: A list of feature selection methods to evaluate.
        thresholds: A dictionary of thresholds for each feature selection method. Defaults to None.

    Returns:
        A dictionary containing the evaluation results for each feature selection method.
    """
    # validate input
    if not os.path.exists(pickled_result_path):
        raise ValueError(f"The file {pickled_result_path} does not exist.")
    if not isinstance(feature_selection_methods, list):
        raise TypeError("feature_selection_methods should be a list.")
    if thresholds is not None and not isinstance(thresholds, dict):
        raise TypeError("thresholds should be a dictionary or None.")

    # unpickle the results
    with open(pickled_result_path, "rb") as file:
        result_dict = pickle.load(file)
    assert isinstance(result_dict, dict)

    # extract cross-validation results
    feature_selection_result_dict = result_dict["method_result_dict"]
    feature_selection_result_cv_list = feature_selection_result_dict["rf_cv_result"]
    assert isinstance(feature_selection_result_cv_list, list)

    # evaluate feature selection results
    result_dict = evaluate_feature_subsets(feature_selection_result_cv_list, feature_selection_methods, thresholds)

    return result_dict


def evaluate_feature_subsets(
    feature_selection_result_cv_list: list,
    feature_selection_methods: list[str],
    thresholds: dict = None,
) -> dict:
    """
    Evaluates feature subsets based on different feature selection methods and thresholds.

    This function iterates over each feature selection method provided, extracts the raw feature selection results,
    and analyzes the feature importance matrix. If a thresholds dictionary is provided, it scales the feature importance
    matrix, applies the threshold specific to the current feature selection method, and analyzes the trimmed feature
    importance matrix. If the feature selection method is "reverse", it applies a p-value to select feature subsets for
    reverse feature selection. It calculates the significant difference of reverse feature selection results with p-values
    based on t-test and Mann-Whitney-U test. It then analyzes the feature importance matrix with and without applying the
    threshold (if provided).

    Args:
        feature_selection_result_cv_list (list): A list of cross-validation results.
        feature_selection_methods (list[str]): A list of feature selection methods to evaluate.
        thresholds (dict, optional): A dictionary of thresholds for each feature selection method. Defaults to None.

    Returns:
        dict: A dictionary containing the evaluation results for each feature selection method.
    """
    result_dict = {}
    for feature_selection_method in feature_selection_methods:
        # extract raw feature selection results
        feature_importance_matrix = extract_feature_importance_matrix(
            feature_selection_result_cv_list, feature_selection_method
        )
        analyze_and_apply_threshold(result_dict, feature_selection_method, feature_importance_matrix, thresholds)

        # apply p-value to select feature subsets for reverse feature selection
        if feature_selection_method == "reverse":
            # calculate significant difference of reverse feature selection results with p-values based on t-test
            # and Mann-Whitney-U test
            for p_value in ["p_values_tt", "p_values_mwu"]:
                feature_importance_matrix = extract_feature_importance_matrix(
                    feature_selection_result_cv_list, feature_selection_method, p_value
                )
                analyze_and_apply_threshold(
                    result_dict,
                    f"{feature_selection_method}_{p_value}",
                    feature_importance_matrix,
                    thresholds,
                )
            # store feature importance matrix and p_values to determine feature weights

    return result_dict


def update_result_dictionary(result_dict: dict, feature_selection_method: str, key: str, value: Any):
    """
    Updates the result dictionary with a given key-value pair.

    Args:
        result_dict: The dictionary to update.
        feature_selection_method: The feature selection method to add to the dictionary.
        key: The key to add to the dictionary.
        value: The value to assign to the key.

    Returns:
        None
    """
    if key not in result_dict.keys():
        result_dict[key] = {}
    result_dict[key][feature_selection_method] = value


def analyze_feature_importance_matrix(
    result_dict: dict, feature_selection_method: str, feature_importance_matrix: np.ndarray
) -> None:
    """
    Analyzes a feature importance matrix and updates a result dictionary with various metrics.

    This function calculates the stability, robustness, subset size, mean subset size, importance ranking,
    and feature weights based on the feature importance matrix. It then updates the result dictionary
    with these metrics.

    Args:
        result_dict: The result dictionary to update.
        feature_selection_method: The feature selection method used.
        feature_importance_matrix: The feature importance matrix to analyze.
    """
    # calculate stability of feature selection
    update_result_dictionary(
        result_dict, feature_selection_method, "stability", stability_estimator.get_stability(feature_importance_matrix)
    )
    # count number of nonzero values per column to determine the robustness
    update_result_dictionary(
        result_dict,
        feature_selection_method,
        "robustness_array",
        np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=0),
    )
    # count number of nonzero values per row to determine the subset size
    update_result_dictionary(
        result_dict,
        feature_selection_method,
        "subset_size_array",
        np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=1),
    )
    # calculate mean subset size
    update_result_dictionary(
        result_dict,
        feature_selection_method,
        "mean_subset_size",
        np.mean(result_dict["subset_size_array"][feature_selection_method], axis=0),
    )
    # calculate importance ranking
    update_result_dictionary(
        result_dict, feature_selection_method, "importance_ranking", np.sum(feature_importance_matrix, axis=0)
    )
    # calculate feature weights
    update_result_dictionary(
        result_dict, feature_selection_method, "feature_weights", calculate_feature_weights(feature_importance_matrix)
    )


def calculate_feature_weights(feature_importance_matrix: np.ndarray):
    # TODO calculate feature weights
    feature_weights = np.sum(feature_importance_matrix, axis=1)
    return feature_weights


def analyze_and_apply_threshold(
    result_dict, feature_selection_method, feature_importance_matrix, thresholds=None
):
    """
    Analyzes a feature importance matrix and applies a threshold if provided.

    This function first evaluates the feature importances, then checks if a threshold is provided. If so, it scales
    the feature importance matrix, applies the threshold, and analyzes the trimmed feature importance matrix.

    Args:
        result_dict: The result dictionary to update.
        feature_selection_method: The feature selection method used.
        feature_importance_matrix: The feature importance matrix to analyze.
        thresholds: A dictionary of thresholds for each feature selection method.
        p_value: The p-value used for reverse feature selection.
    """
    # copy feature importance matrix
    feature_importance_matrix_cp = feature_importance_matrix.copy()

    # TODO immer skalieren?
    analyze_feature_importance_matrix(result_dict, feature_selection_method, feature_importance_matrix)

    if thresholds is not None:
        # trimmed_feature_importance_matrix = apply_threshold(
        #     scale_feature_importance_matrix(feature_importance_matrix, MinMaxScaler()),
        #     thresholds[feature_selection_method],
        # )
        print("method= ", feature_selection_method)
        trimmed_feature_importance_matrix, threshold = optimize_threshold(feature_importance_matrix)
        analyze_feature_importance_matrix(
            result_dict,
            f"{feature_selection_method}_threshold_{threshold}",
            trimmed_feature_importance_matrix,
        )
    assert np.array_equal(
        feature_importance_matrix, feature_importance_matrix_cp
    ), "feature importance matrix was changed"


def scale_feature_importance_matrix(feature_importance_matrix: np.ndarray, scaler=None):
    # check if the sum of all columns is one or zero
    if np.all(
        np.isclose(np.sum(feature_importance_matrix, axis=1), 1)
        | np.isclose(np.sum(feature_importance_matrix, axis=1), 0)
    ):
        return feature_importance_matrix

    # transform feature importance matrix to have all columns sum up to one or zero
    transformed_importance_matrix = feature_importance_matrix.copy()
    sum_of_all_importances = np.sum(feature_importance_matrix, axis=1)
    for i, importance_sum in enumerate(sum_of_all_importances):
        if importance_sum > 0:
            transformed_importance_matrix[:, i] = feature_importance_matrix[:, i] / importance_sum

    assert np.all(
        np.isclose(np.sum(transformed_importance_matrix, axis=1), 1)
        | np.isclose(np.sum(transformed_importance_matrix, axis=1), 0)
    ), np.sum(transformed_importance_matrix, axis=1)
    return transformed_importance_matrix


def apply_threshold(feature_importance_matrix: np.ndarray, threshold: float):
    if threshold is None:
        raise ValueError("Threshold must not be None.")

    # # check if threshold is in the desired range
    # assert (
    #     feature_importance_matrix.max() > threshold > feature_importance_matrix.min()
    # ), f"Threshold is not in the desired range: max {feature_importance_matrix.max()} > threshold {threshold} > min {feature_importance_matrix.min()}"

    # preserve the original feature importance matrix
    feature_importance_matrix_cp = feature_importance_matrix.copy()

    # apply threshold and set all values below threshold to zero
    feature_importance_matrix_cp[feature_importance_matrix_cp < threshold] = 0
    assert feature_importance_matrix_cp.min() == 0 or feature_importance_matrix_cp.min() >= threshold, str(
        feature_importance_matrix_cp.min()
    )
    return feature_importance_matrix_cp


def extract_feature_importance_matrix(
    feature_selection_result_cv_list: list,
    method: str,
    p_value: str = None,
    threshold: float = None,
):
    # extract feature selection matrix
    feature_importance_matrix = np.empty(
        (len(feature_selection_result_cv_list), feature_selection_result_cv_list[0].shape[0])
    )

    # iterate over cross-validation results
    for i, cv_iteration_result_df in enumerate(feature_selection_result_cv_list):
        # _normalize_feature_subsets(cv_iteration_result_pd)
        feature_importance_matrix[i, :] = get_selected_feature_subset(
            cv_iteration_result_df, method, p_value, threshold
        ).values

    # check of each row contains at least one nonzero value
    assert np.all(
        np.sum(feature_importance_matrix, axis=1) > 0
    ), f"{np.sum(feature_importance_matrix, axis=1)} {method}"

    # check the shape of the feature importance matrix
    assert feature_importance_matrix.shape == (
        len(feature_selection_result_cv_list),
        feature_selection_result_cv_list[0].shape[0],
    ), feature_importance_matrix.shape

    return feature_importance_matrix


def get_selected_feature_subset(
    cv_fold_result_df: pd.DataFrame, method: str, p_value: str = None, threshold: float = None
) -> pd.Series:
    if method == "standard":
        assert p_value is None
        feature_subset_series = cv_fold_result_df["standard"]
        # sum of all values in the feature subset series must be one
        res = feature_subset_series.sum()
        assert np.isclose(feature_subset_series.sum(), 1.0), feature_subset_series.sum()
    elif method == "standard_shap":
        assert p_value is None
        feature_subset_series = cv_fold_result_df["shap_values"]
        # TODO: sum of all values in the feature subset series must be one?

    elif method == "reverse":
        feature_subset_series = get_feature_subset_for_reverse_feature_selection(cv_fold_result_df, p_value)
    else:
        raise ValueError("Unknown feature_selection_method.")

    if threshold is not None:
        # apply threshold
        feature_subset_series = feature_subset_series >= threshold

    assert feature_subset_series.sum() > 0, feature_subset_series.sum()

    return feature_subset_series


def get_feature_subset_for_reverse_feature_selection(cv_fold_result_df: pd.DataFrame, p_value: str = None) -> pd.Series:
    # replace infinite values with zero
    cv_fold_result_df["labeled"] = cv_fold_result_df["labeled"].replace([np.inf, -np.inf], 0)
    cv_fold_result_df["unlabeled"] = cv_fold_result_df["unlabeled"].replace([np.inf, -np.inf], 0)

    assert not cv_fold_result_df["labeled"].isnull().values.any(), cv_fold_result_df["labeled"].isnull().sum()
    assert not cv_fold_result_df["unlabeled"].isnull().values.any(), cv_fold_result_df["unlabeled"].isnull().sum()

    # extract feature subset for reverse feature selection based on the percentage difference
    # between labeled and unlabeled OOB error
    selected_feature_subset = (cv_fold_result_df["unlabeled"] - cv_fold_result_df["labeled"]) / cv_fold_result_df[
        "unlabeled"
    ]
    # replace nan values with zero
    selected_feature_subset = selected_feature_subset.fillna(0)
    # check if the feature subset contains nan values
    assert not selected_feature_subset.isnull().values.any(), selected_feature_subset.isnull().sum()

    # check if the feature subset contains infinite values
    assert not np.isinf(selected_feature_subset).values.any(), selected_feature_subset.isinf().sum()

    # check if the feature subset contains negative values
    assert not (selected_feature_subset < 0).values.any(), selected_feature_subset[selected_feature_subset < 0]

    # if "percent" not in cv_fold_result_df.columns:
    cv_fold_result_df.insert(2, f"percent_{p_value}", selected_feature_subset, allow_duplicates=False)

    if p_value is not None:
        # apply p-value threshold
        for feature in selected_feature_subset.index:
            if cv_fold_result_df[p_value][feature] >= 0.05:
                selected_feature_subset[feature] = 0
            elif not math.isnan(cv_fold_result_df[p_value][feature]):
                # print(-np.log(cv_fold_result_df[p_value][feature]))
                pass

    return selected_feature_subset


def optimize_threshold(feature_importance_matrix):
    def objective(trial):
        threshold = trial.suggest_uniform("threshold", 0.0, 1.0)
        trimmed_feature_importance_matrix = apply_threshold(feature_importance_matrix, threshold)
        return stability_estimator.get_stability(trimmed_feature_importance_matrix)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print(study.best_params)
    return apply_threshold(feature_importance_matrix, study.best_params["threshold"]), study.best_params["threshold"]


thresholds_dict = {"standard": 0.1, "standard_shap": 0.002, "reverse": 0.05}
# thresholds = None
result_dict_final = evaluate(
    "/home/sigrun/PycharmProjects/reverse_feature_selection/results/prostate_loo_result_dict.pkl",
    feature_selection_methods=["standard", "standard_shap", "reverse"],
    thresholds=thresholds_dict,
)
for key, value in result_dict_final["stability"].items():
    print(key, ":", value)
    print("mean subset size: ", result_dict_final["mean_subset_size"][key])
    print("  ")
