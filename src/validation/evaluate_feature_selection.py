import math
import pickle
from typing import Literal, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

import stability_estimator


def evaluate(pickled_result_path: str, feature_selection_methods: list[str], thresholds: dict = None):
    # unpickle the results
    with open(pickled_result_path, "rb") as file:
        result_dict = pickle.load(file)

    # extract feature selection result dict
    feature_selection_result_dict = result_dict["method_result_dict"]

    # extract feature selection result cross-validation results
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
        feature_importance_matrix = _extract_feature_importance_matrix(
            feature_selection_result_cv_list, feature_selection_method
        )
        analyze_and_apply_threshold(result_dict, feature_selection_method, feature_importance_matrix, thresholds)

        # _analyze_feature_importance_matrix(result_dict, feature_selection_method, feature_importance_matrix)
        #
        # if thresholds is not None:
        #     trimmed_feature_importance_matrix = apply_threshold(
        #         scale_feature_importance_matrix(feature_importance_matrix, MinMaxScaler()),
        #         thresholds[feature_selection_method],
        #     )
        #     _analyze_feature_importance_matrix(
        #         result_dict,
        #         f"{feature_selection_method}_threshold_{thresholds[feature_selection_method]}",
        #         trimmed_feature_importance_matrix,
        #     )

        # apply p-value to select feature subsets for reverse feature selection
        if feature_selection_method == "reverse":
            # calculate significant difference of reverse feature selection results with p-values based on t-test
            # and Mann-Whitney-U test
            for p_value in ["p_values_tt", "p_values_mwu"]:
                feature_importance_matrix = _extract_feature_importance_matrix(
                    feature_selection_result_cv_list, feature_selection_method, p_value
                )
                # feature_importance_matrix_copy = feature_importance_matrix.copy()
                # _analyze_feature_importance_matrix(
                #     result_dict, f"{feature_selection_method}_{p_value}", feature_importance_matrix
                # )
                # assert np.array_equal(
                #     feature_importance_matrix, feature_importance_matrix_copy
                # ), "feature importance matrix was changed"
                #
                # if thresholds is not None:
                #     feature_importance_matrix = apply_threshold(
                #         scale_feature_importance_matrix(feature_importance_matrix, MinMaxScaler()),
                #         thresholds[feature_selection_method],
                #     )
                #     _analyze_feature_importance_matrix(
                #         result_dict,
                #         f"{feature_selection_method}_{p_value}_threshold_{thresholds[feature_selection_method]}",
                #         feature_importance_matrix,
                #     )
                analyze_and_apply_threshold(
                    result_dict,
                    f"{feature_selection_method}_{p_value}",
                    feature_importance_matrix,
                    thresholds,
                )
    return result_dict


def _analyze_feature_importance_matrix(
    result_dict: dict, feature_selection_method: str, feature_importance_matrix: np.ndarray
):
    # calculate stability of feature selection
    if "stability" not in result_dict.keys():
        result_dict["stability"] = {}
    result_dict["stability"][feature_selection_method] = stability_estimator.get_stability(feature_importance_matrix)

    # count number of nonzero values per column to determine the robustness
    if "robustness_array" not in result_dict.keys():
        result_dict["robustness_array"] = {}
    result_dict["robustness_array"][feature_selection_method] = np.sum(
        np.where(feature_importance_matrix > 0, 1, 0), axis=0
    )

    # count number of nonzero values per row to determine the subset size
    if "subset_size_array" not in result_dict.keys():
        result_dict["subset_size_array"] = {}
    result_dict["subset_size_array"][feature_selection_method] = np.sum(
        np.where(feature_importance_matrix > 0, 1, 0), axis=1
    )

    # calculate mean subset size
    if "mean_subset_size" not in result_dict.keys():
        result_dict["mean_subset_size"] = {}
    result_dict["mean_subset_size"][feature_selection_method] = np.mean(
        result_dict["subset_size_array"][feature_selection_method], axis=0
    )

    # calculate importance ranking
    if "importance_ranking" not in result_dict.keys():
        result_dict["importance_ranking"] = {}
    result_dict["importance_ranking"][feature_selection_method] = np.sum(feature_importance_matrix, axis=0)

    # calculate feature weights
    if "feature_weights" not in result_dict.keys():
        result_dict["feature_weights"] = {}
    result_dict["feature_weights"][feature_selection_method] = calculate_feature_weights(feature_importance_matrix)


def calculate_feature_weights(feature_importance_matrix: np.ndarray):
    # TODO calculate feature weights
    feature_weights = np.sum(feature_importance_matrix, axis=1)
    return feature_weights


def analyze_and_apply_threshold(result_dict, feature_selection_method, feature_importance_matrix, thresholds=None):
    _analyze_feature_importance_matrix(result_dict, feature_selection_method, feature_importance_matrix)

    if thresholds is not None and feature_selection_method in thresholds:
        trimmed_feature_importance_matrix = apply_threshold(
            scale_feature_importance_matrix(feature_importance_matrix, MinMaxScaler()),
            thresholds[feature_selection_method],
        )
        _analyze_feature_importance_matrix(
            result_dict,
            f"{feature_selection_method}_threshold_{thresholds[feature_selection_method]}",
            trimmed_feature_importance_matrix,
        )


def scale_feature_importance_matrix(feature_importance_matrix: np.ndarray, scaler):
    # scale feature importance matrix
    transformed_importance_matrix = scaler.fit_transform(feature_importance_matrix)
    assert transformed_importance_matrix.min() >= 0, transformed_importance_matrix.min()
    assert (
        math.isclose(transformed_importance_matrix.max(), 1) or transformed_importance_matrix.max() < 1
    ), transformed_importance_matrix.max()

    # check if original matrix needs to be transformed or was already in the desired range
    if not (feature_importance_matrix.min() >= 0, feature_importance_matrix.min()) and (
        feature_importance_matrix.max() <= 1,
        feature_importance_matrix.max(),
    ):
        # check if the feature importance matrix was transformed
        assert not np.array_equal(
            feature_importance_matrix, transformed_importance_matrix
        ), "feature importance matrix was not transformed"

    return transformed_importance_matrix


def apply_threshold(feature_importance_matrix: np.ndarray, threshold: float):
    if threshold is None:
        raise ValueError("Threshold must not be None.")

    # check if threshold is in the desired range
    assert (
        feature_importance_matrix.max() > threshold > feature_importance_matrix.min()
    ), f"Threshold is not in the desired range: max {feature_importance_matrix.max()} > threshold {threshold} > min {feature_importance_matrix.min()}"

    # preserve the original feature importance matrix
    feature_importance_matrix_cp = feature_importance_matrix.copy()

    # apply threshold and set all values below threshold to zero
    feature_importance_matrix_cp[feature_importance_matrix_cp < threshold] = 0
    assert feature_importance_matrix_cp.min() == 0 or feature_importance_matrix_cp.min() >= threshold, str(
        feature_importance_matrix_cp.min()
    )
    return feature_importance_matrix_cp


def _extract_feature_importance_matrix(
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
        feature_importance_matrix[i, :] = _get_selected_feature_subset(
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


def _get_selected_feature_subset(
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
        feature_subset_series = _get_feature_subset_for_reverse_feature_selection(cv_fold_result_df, p_value)
    else:
        raise ValueError("Unknown feature_selection_method.")

    if threshold is not None:
        # apply threshold
        feature_subset_series = feature_subset_series >= threshold

    assert feature_subset_series.sum() > 0, feature_subset_series.sum()

    return feature_subset_series


def _get_feature_subset_for_reverse_feature_selection(
    cv_fold_result_df: pd.DataFrame, p_value: str = None
) -> pd.Series:
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


thresholds_dict = {"standard": 0.1, "standard_shap": 0.002, "reverse": 0.05}
# thresholds = None
result_dict_final = evaluate(
    "/home/sigrun/PycharmProjects/reverse_feature_selection/results/leukemia_loo_result_dict.pkl",
    feature_selection_methods=["standard", "standard_shap", "reverse"],
    thresholds=thresholds_dict,
)
print(result_dict_final)
