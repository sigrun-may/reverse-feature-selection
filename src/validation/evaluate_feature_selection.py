import math
import os
import pickle
from typing import Any, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    log_loss,
    brier_score_loss,
)
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

import stability_estimator
from src.reverse_feature_selection.data_loader_tools import load_data_with_standardized_sample_size
from weighted_manhattan_distance import WeightedManhattanDistance


def evaluate(
    data_df: pd.DataFrame, pickled_result_path: str, feature_selection_methods: list[str], thresholds: dict = None
) -> Tuple[dict, dict]:
    """
    Evaluates the feature selection results.

    This function loads the pickled results from the specified path, extracts the feature selection results,
    and evaluates them based on the provided feature selection methods and thresholds.

    Args:
        data_df: The data to evaluate.
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
        raw_result_dict = pickle.load(file)
    assert isinstance(raw_result_dict, dict)

    # extract cross-validation results
    feature_selection_result_dict = raw_result_dict["method_result_dict"]
    feature_selection_result_cv_list = feature_selection_result_dict["rf_cv_result"]
    cross_validation_indices_list = raw_result_dict["indices"]
    assert isinstance(feature_selection_result_cv_list, list)
    assert isinstance(cross_validation_indices_list, list)

    # evaluate feature selection results
    subset_evaluation_result_dict = evaluate_feature_subsets(
        feature_selection_result_cv_list, feature_selection_methods, thresholds
    )
    performance_evaluation_result_dict = evaluate_performance(
        data_df, cross_validation_indices_list, subset_evaluation_result_dict
    )
    return subset_evaluation_result_dict, performance_evaluation_result_dict


def calculate_auc_auprc(y_true, y_predict):
    """
    Calculates the area under the ROC curve and the area under the precision-recall curve.

    Args:
        y_true: The true labels.
        y_predict: The predicted labels.

    Returns:
        The area under the ROC curve and the area under the precision-recall curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    auc_value = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_true, y_predict)
    auprc = average_precision_score(y_true, y_predict)
    return auc_value, auprc, fpr, tpr, thresholds


def evaluate_performance(
    data_df: pd.DataFrame, cross_validation_indices_list: list, subset_evaluation_result_dict: dict, k: int = 5
) -> dict:
    """
    Evaluates the performance of the feature selection methods.

    This function calculates the performance of the feature selection methods based on the cross-validation indices
    and the subset evaluation results. It then updates the subset evaluation results with the performance results.

    Args:
        data_df: The data to evaluate.
        cross_validation_indices_list: A list of cross-validation indices.
        subset_evaluation_result_dict: A dictionary containing the evaluation results for each feature selection method.
        k: The number of neighbors to use. Defaults to 5.
    """
    performance_evaluation_result_dict = {"performance": {}}
    # iterate over cross-validation indices
    for i, (train_indices, test_indices) in enumerate(cross_validation_indices_list):
        # extract training and test data samples
        train_data_df = data_df.iloc[train_indices, :]
        test_data_df = data_df.iloc[test_indices, :]

        # extract and transform x and y values
        x_train = train_data_df.iloc[:, 1:].values
        y_train = train_data_df.iloc[:, 0].values
        x_test = test_data_df.iloc[:, 1:].values
        y_test = test_data_df.iloc[:, 0].values

        # iterate over feature selection methods rankings
        for feature_selection_method in subset_evaluation_result_dict["importance_ranking"].keys():
            # extract feature subset
            feature_subset = subset_evaluation_result_dict["importance_ranking"][feature_selection_method][i]
            assert feature_subset.size == train_data_df.shape[1] - 1, feature_subset.size  # -1 for label column

            # train and evaluate the model
            knn = KNeighborsClassifier(
                n_neighbors=k, metric=WeightedManhattanDistance(feature_subset), algorithm="brute", n_jobs=-1
            )
            knn.fit(x_train, y_train)
            predicted_proba_y = knn.predict_proba(x_test)[0]
            predicted_y = knn.predict(x_test)[0]

            # initialize list in feature_selection_method_performance_evaluation_dict
            # for current feature_selection_method
            if f"{feature_selection_method}_predict" not in performance_evaluation_result_dict.keys():
                performance_evaluation_result_dict[f"{feature_selection_method}_predict"] = []
            if f"{feature_selection_method}_predict_proba" not in performance_evaluation_result_dict.keys():
                performance_evaluation_result_dict[f"{feature_selection_method}_predict_proba"] = []
            if f"{feature_selection_method}_true" not in performance_evaluation_result_dict.keys():
                performance_evaluation_result_dict[f"{feature_selection_method}_true"] = []

            # append score to list in feature_selection_method_performance_evaluation_dict
            performance_evaluation_result_dict[f"{feature_selection_method}_predict"].append(predicted_y)
            performance_evaluation_result_dict[f"{feature_selection_method}_predict_proba"].append(predicted_proba_y)
            performance_evaluation_result_dict[f"{feature_selection_method}_true"].append(y_test[0])

    # calculate performance metrics
    for feature_selection_method in subset_evaluation_result_dict["importance_ranking"].keys():
        # calculate performance metrics
        accuracy = accuracy_score(
            performance_evaluation_result_dict[f"{feature_selection_method}_true"],
            performance_evaluation_result_dict[f"{feature_selection_method}_predict"],
        )
        precision = precision_score(
            performance_evaluation_result_dict[f"{feature_selection_method}_true"],
            performance_evaluation_result_dict[f"{feature_selection_method}_predict"],
        )
        recall = recall_score(
            performance_evaluation_result_dict[f"{feature_selection_method}_true"],
            performance_evaluation_result_dict[f"{feature_selection_method}_predict"],
        )
        f1 = f1_score(
            performance_evaluation_result_dict[f"{feature_selection_method}_true"],
            performance_evaluation_result_dict[f"{feature_selection_method}_predict"],
        )
        auc, auprc, fpr, tpr, thresholds = calculate_auc_auprc(
            performance_evaluation_result_dict[f"{feature_selection_method}_true"],
            performance_evaluation_result_dict[f"{feature_selection_method}_predict"],
        )
        # log loss
        log_loss_value = log_loss(
            performance_evaluation_result_dict[f"{feature_selection_method}_true"],
            performance_evaluation_result_dict[f"{feature_selection_method}_predict_proba"],
        )
        # brier score loss
        # extract the probability for the positive class
        probability_positive_class = np.array(
            [proba[1] for proba in performance_evaluation_result_dict[f"{feature_selection_method}_predict_proba"]]
        )
        brier_score_loss_value = brier_score_loss(
            performance_evaluation_result_dict[f"{feature_selection_method}_true"], probability_positive_class
        )

        # initialize dictionary for performance metrics
        performance_evaluation_result_dict["performance"][feature_selection_method] = {}

        # add performance metrics to result dictionary
        performance_evaluation_result_dict["performance"][feature_selection_method]["accuracy"] = accuracy
        performance_evaluation_result_dict["performance"][feature_selection_method]["precision"] = precision
        performance_evaluation_result_dict["performance"][feature_selection_method]["recall"] = recall
        performance_evaluation_result_dict["performance"][feature_selection_method]["f1"] = f1
        performance_evaluation_result_dict["performance"][feature_selection_method]["auc"] = auc
        performance_evaluation_result_dict["performance"][feature_selection_method]["auprc"] = auprc
        performance_evaluation_result_dict["performance"][feature_selection_method]["fpr"] = fpr
        performance_evaluation_result_dict["performance"][feature_selection_method]["tpr"] = tpr
        performance_evaluation_result_dict["performance"][feature_selection_method]["thresholds"] = thresholds
        performance_evaluation_result_dict["performance"][feature_selection_method]["log_loss"] = log_loss_value
        performance_evaluation_result_dict["performance"][feature_selection_method][
            "brier_score_loss"
        ] = brier_score_loss_value

    # # plot metrics of different feature selection methods with seaborn
    # for feature_selection_method in performance_evaluation_result_dict["performance"].keys():
    #     # plot roc curve
    #     plt.figure()
    #     lw = 2
    #     plt.plot(
    #         performance_evaluation_result_dict["performance"][feature_selection_method]["fpr"],
    #         performance_evaluation_result_dict["performance"][feature_selection_method]["tpr"],
    #         color="darkorange",
    #         lw=lw,
    #         label=f"ROC curve (area = {performance_evaluation_result_dict['performance'][feature_selection_method]['auc']:.2f})",
    #     )
    #     plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    #     plt.xlim([0.0, 1.05])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title(f"ROC curve for {feature_selection_method}")
    #     plt.legend(loc="lower right")
    #     # plt.savefig(f"/home/sigrun/PycharmProjects/reverse_feature_selection/results/roc_curve_{feature_selection_method}.png")
    #     # plt.show()
    #
    #     # plot precision-recall curve
    #     plt.figure()
    #     lw = 2
    #     plt.plot(
    #         performance_evaluation_result_dict["performance"][feature_selection_method]["recall"],
    #         performance_evaluation_result_dict["performance"][feature_selection_method]["precision"],
    #         color="darkorange",
    #         lw=lw,
    #         label=f"PR curve (area = {performance_evaluation_result_dict['performance'][feature_selection_method]['auprc']:.2f})",
    #     )
    #     plt.xlim([0.0, 1.05])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.title(f"PR curve for {feature_selection_method}")
    #     plt.legend(loc="lower right")
    #     # plt.savefig(f"/home/sigrun/PycharmProjects/reverse_feature_selection/pr_curve_{feature_selection_method}.png")
    #     # plt.show()
    # # plot f1 score
    # f1_score_df = pd.DataFrame(
    #     {
    #         # "feature_selection_method": list(performance_evaluation_result_dict["performance"].keys()),
    #         "f1_score": [value["f1"] for value in performance_evaluation_result_dict["performance"].values()],
    #     },
    #     index=list(performance_evaluation_result_dict["performance"].keys()),
    # )
    # # seaborn barchart based on f1_score_series
    # sns.set_theme(style="whitegrid")
    # sns.set(font_scale=1)
    # sns.set(rc={"figure.figsize": (11.7, 12.0)})
    # ax = sns.barplot(x=f1_score_df.index, y=f1_score_df["f1_score"])
    # ax.set_xticks(ax.get_xticks(), labels=list(f1_score_df.index), rotation=45, horizontalalignment="right")
    # ax.set(xlabel=list(f1_score_df.index), ylabel="F1 Score")
    # plt.savefig("/home/sigrun/PycharmProjects/reverse_feature_selection/f1_score.png")
    # plt.show()

    # sns.set_theme(style="whitegrid")
    # sns.set(rc={"figure.figsize": (11.7, 8.27)})
    # sns.set(font_scale=1.5)
    # ax = sns.barplot(data=f1_score_df, y="f1_score")
    # # set xticks to be rotated
    # # ax.set_xticks(ax.get_xticks(), labels=f1_score_df["feature_selection_method"], rotation=45, horizontalalignment="right")
    # ax.set(xlabel="Feature Selection Method", ylabel="F1 Score")
    # # plt.savefig("/home/sigrun/PycharmProjects/reverse_feature_selection/results/f1_score.png")
    # plt.show()

    # # plot auc
    # auc_df = pd.DataFrame(
    #     {
    #         "feature_selection_method": list(performance_evaluation_result_dict["performance"].keys()),
    #         "auc": [value["auc"] for value in performance_evaluation_result_dict["performance"].values()],
    #     }
    # )
    # sns.set_theme(style="whitegrid")
    # # sns.set(rc={"figure.figsize": (21.7, 18.27)})
    # ax = auc_df.plot.bar(x="feature_selection_method", y="auc")
    # ax.set(xlabel=auc_df["feature_selection_method"].values, ylabel="AUC")
    # plt.savefig("/home/sigrun/PycharmProjects/reverse_feature_selection/auc.png")
    # plt.show()

    return performance_evaluation_result_dict


def feature_weighted_knn(
    train_data: pd.DataFrame, test_data: pd.DataFrame, feature_weights_cv_list: list[pd.Series], k: int
):
    """
    Calculates the feature weighted k-nearest neighbors.

    Args:
        train_data: The training data.
        test_data: The test data.
        feature_weights_cv_list: A list of feature weights for each cross-validation result.
        k: The number of neighbors to use.
    """
    # calculate feature weighted k-nearest neighbors
    # iterate over cross-validation results
    for feature_weights_series in feature_weights_cv_list:
        pass

    pass


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
    importance matrix. If the feature selection method is "reverse", it applies a p-metric_value to select feature subsets for
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

        # apply p-metric_value to select feature subsets for reverse feature selection
        if feature_selection_method == "reverse":
            # calculate significant difference of reverse feature selection results with p-values based on t-test
            # and Mann-Whitney-U test
            for p_value in ["p_values_tt", "p_values_mwu"]:
                feature_importance_matrix = extract_feature_importance_matrix(
                    feature_selection_result_cv_list, feature_selection_method, p_value
                )
                analyze_and_apply_threshold(
                    result_dict,
                    f"{feature_selection_method}_fraction_{p_value}",
                    feature_importance_matrix,
                    thresholds,
                )
                # apply negative log p-values as feature importances
                apply_log_p_value_as_feature_importance(
                    result_dict,
                    f"{feature_selection_method}_log_{p_value}",
                    feature_selection_result_cv_list,
                    p_value,
                )
    return result_dict


def apply_log_p_value_as_feature_importance(
    result_dict: dict,
    feature_selection_method: str,
    feature_selection_result_cv_list: list,
    p_value: str,
):
    """
    Applies the negative log of the p-value as feature importance.

    This function iterates over each cross-validation result, extracts the p-values, and calculates the negative log
    of the p-value. It then updates the feature selection result with the negative log of the p-value as feature
    importance.

    Args:
        result_dict: The result dictionary to update.
        feature_selection_method: The feature selection method used.
        feature_selection_result_cv_list: A list of cross-validation results.
        p_value: The p-value to extract. Either "p_values_tt" for T-test
                 or "p_values_mwu" for Mann–Whitney U test.
    """
    p_value_importance_list = []
    # iterate over cross-validation results
    for cv_iteration_result_df in feature_selection_result_cv_list:
        # extract p-values
        p_values_array = cv_iteration_result_df[p_value].values
        # replace nan values with zero
        p_values_array = np.nan_to_num(p_values_array)
        # set the p-values smaller than 0.05 to zero
        p_values_array[p_values_array < 0.05] = 0
        # calculate negative log of the nonzero p-values
        p_values_array[p_values_array > 0] = -np.log(p_values_array[p_values_array > 0])
        p_value_importance_list.append(p_values_array)

    # convert list of arrays to matrix
    p_value_importance_matrix = np.vstack(p_value_importance_list)
    assert p_value_importance_matrix.shape == (
        len(feature_selection_result_cv_list),
        feature_selection_result_cv_list[0].shape[0],
    ), p_value_importance_matrix.shape

    # update result dictionary with p_value feature importance matrix
    result_dict["importance_ranking"][feature_selection_method] = p_value_importance_matrix


def update_result_dictionary(result_dict: dict, feature_selection_method: str, metric_name: str, metric_value: Any):
    """
    Updates the result dictionary with a given metric_name-value pair.

    Args:
        result_dict: The dictionary to update.
        feature_selection_method: The feature selection method to add to the dictionary.
        metric_name: The metric_name to add to the dictionary.
        metric_value: The value to assign to the metric_name.

    Returns:
        None
    """
    if metric_name not in result_dict.keys():
        result_dict[metric_name] = {}
    result_dict[metric_name][feature_selection_method] = metric_value


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
    update_result_dictionary(result_dict, feature_selection_method, "importance_ranking", feature_importance_matrix)


def calculate_feature_weights(feature_selection_method: str, feature_importance_matrix: np.ndarray):
    # TODO calculate feature weights
    # oob improvement fraction
    # check robustness
    feature_weights = np.sum(feature_importance_matrix, axis=1)
    return feature_weights


def analyze_and_apply_threshold(result_dict, feature_selection_method, feature_importance_matrix, thresholds=None):
    """
    Analyzes a feature importance matrix and applies a threshold if provided.

    This function first evaluates the feature importances, then checks if a threshold is provided. If so, it scales
    the feature importance matrix, applies the threshold, and analyzes the trimmed feature importance matrix.

    Args:
        result_dict: The result dictionary to update.
        feature_selection_method: The feature selection method used.
        feature_importance_matrix: The feature importance matrix to analyze.
        thresholds: A dictionary of thresholds for each feature selection method.
    """
    analyze_feature_importance_matrix(result_dict, feature_selection_method, feature_importance_matrix)

    if thresholds is not None:
        print("method= ", feature_selection_method)
        trimmed_feature_importance_matrix, threshold = optimize_threshold(feature_importance_matrix)
        analyze_feature_importance_matrix(
            result_dict,
            f"{feature_selection_method}_threshold_{threshold}",
            trimmed_feature_importance_matrix,
        )


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
    if not feature_importance_matrix.max() > threshold > feature_importance_matrix.min():
        # return matrix of zeros if threshold is not in the desired range
        return np.zeros(feature_importance_matrix.shape)

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

    # check of each row contains at least one nonzero metric_value
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
        # apply p-metric_value threshold
        for feature in selected_feature_subset.index:
            if cv_fold_result_df[p_value][feature] >= 0.05:
                selected_feature_subset[feature] = 0
            elif not math.isnan(cv_fold_result_df[p_value][feature]):
                # print(-np.log(cv_fold_result_df[p_value][feature]))
                pass

    return selected_feature_subset


def optimize_threshold(feature_importance_matrix):
    def objective(trial):
        threshold = trial.suggest_float(
            "threshold", np.min(feature_importance_matrix), np.max(feature_importance_matrix) - 0.000001
        )
        trimmed_feature_importance_matrix = apply_threshold(feature_importance_matrix, threshold)
        return stability_estimator.get_stability(trimmed_feature_importance_matrix)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print(study.best_params)
    return apply_threshold(feature_importance_matrix, study.best_params["threshold"]), study.best_params["threshold"]


thresholds_dict = {"standard": 0.1, "standard_shap": 0.002, "reverse": 0.05}
# thresholds = None
input_data_df = load_data_with_standardized_sample_size("prostate")
subset_result_dict_final, performance_result_dict_final = evaluate(
    input_data_df,
    "/home/sigrun/PycharmProjects/reverse_feature_selection/results/prostate_loo_result_dict.pkl",
    feature_selection_methods=["standard", "standard_shap", "reverse"],
    thresholds=thresholds_dict,
)
for key, value in subset_result_dict_final["stability"].items():
    print(key, ":", value)
    print("mean subset size: ", subset_result_dict_final["mean_subset_size"][key])
    print("  ")

# iterate over all performance metrics
for performance_metric in performance_result_dict_final["performance"]["standard"].keys():
    print(performance_metric)
    if performance_metric == "fpr" or performance_metric == "tpr" or performance_metric == "thresholds":
        continue

    # Extract the performance metric, stability, and number of selected features
    data = []
    for method in performance_result_dict_final["performance"].keys():
        performance = performance_result_dict_final["performance"][method][performance_metric]
        importance_matrix = subset_result_dict_final["importance_ranking"][method]
        stability = stability_estimator.get_stability(importance_matrix)
        subset_sizes = np.sum(np.where(importance_matrix > 0, 1, 0), axis=1)
        num_features = np.mean(subset_sizes, axis=0)
        data.append([method, performance, stability, num_features])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["method", "performance", "stability", "num_features"])

    # # Plot
    # plt.figure(figsize=(10, 26))
    # scatter = plt.scatter(df["num_features"], df["performance"], c=df["stability"], cmap='viridis')
    # plt.colorbar(scatter, label="Stability")
    # plt.xlabel("Number of Selected Features")
    # plt.ylabel("Performance Metric")
    # plt.title("Performance vs Number of Selected Features (colored by Stability)")
    # plt.savefig("/home/sigrun/PycharmProjects/reverse_feature_selection/results/feature_selection_performance.png")
    # plt.show()

    # # Iterate over unique methods
    # for method in df["method"].unique():
    #     # Create a scatter plot for each method
    #     subset_df = df[df["method"] == method]
    #     scatter = plt.scatter(subset_df["num_features"], subset_df["performance"], c=subset_df["stability"], cmap='viridis', label=method)
    # plt.figure(figsize=(10, 26))
    # plt.colorbar(scatter, label="Stability")
    # plt.xlabel("Number of Selected Features")
    # plt.ylabel("Performance Metric")
    # plt.title("Performance vs Number of Selected Features (colored by Stability)")
    # # add legend
    # handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    # legend1 = plt.legend(handles, labels, loc="upper right", title="Stability")
    # plt.gca().add_artist(legend1)
    # plt.show()

    # Create a new figure
    plt.figure(figsize=(24, 8))

    # Iterate over unique methods
    for method in df["method"].unique():
        # Create a scatter plot for each method
        subset_df = df[df["method"] == method]
        scatter = plt.scatter(
            subset_df["num_features"], subset_df["performance"], c=subset_df["stability"], cmap="viridis", label=method
        )

    # Add colorbar, labels, and title
    plt.colorbar(scatter, label="Stability")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Performance Metric")
    plt.title(f"{performance_metric} vs Number of Selected Features (colored by Stability)")

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Methods", bbox_to_anchor=(1.2, 1), loc="upper left", borderaxespad=0)

    # Save the figure
    plt.savefig(
        f"/home/sigrun/PycharmProjects/reverse_feature_selection/feature_selection_performance_{performance_metric}.png"
    )

    # convert the figure to plotly
    fig = px.scatter(df, x="num_features", y="performance", color="stability", hover_data=["method"])
    fig.update_layout(
        title=f"{performance_metric} vs Number of Selected Features (colored by Stability)",
        xaxis_title="Number of Selected Features",
        yaxis_title=performance_metric,
        legend_title="Stability",
    )
    fig.show()
