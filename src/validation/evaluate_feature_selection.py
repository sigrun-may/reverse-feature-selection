# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Evaluation of feature selection results."""

import math
import os
import pickle
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import stability_estimator
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

from src.reverse_feature_selection.data_loader_tools import load_data_with_standardized_sample_size


def evaluate(data_df: pd.DataFrame, pickled_result_path: str, plot=False) -> Tuple[dict, dict]:
    """
    Evaluate the feature selection results.

    This function loads the pickled results from the specified path, extracts the feature selection results,
    and evaluates them for each provided feature selection method.

    Args:
        data_df: The data to evaluate.
        pickled_result_path: The path to the pickled results file.
        feature_selection_methods: A list of feature selection methods to evaluate.
        plot: A boolean indicating whether to plot the results. Defaults to False.


    Returns:
        A dictionary containing the evaluation results for each feature selection method.
    """
    # validate input
    if not os.path.exists(pickled_result_path):
        raise ValueError(f"The file {pickled_result_path} does not exist.")

    # unpickle the results
    with open(pickled_result_path, "rb") as file:
        raw_result_dict = pickle.load(file)
    assert isinstance(raw_result_dict, dict)

    # evaluate feature selection results
    subset_evaluation_result_dict = evaluate_feature_subsets(raw_result_dict)

    # evaluate performance
    performance_evaluation_result_dict = evaluate_performance(
        data_df, raw_result_dict["indices"], subset_evaluation_result_dict, k=7
    )
    if plot:
        plot_performance(performance_evaluation_result_dict, subset_evaluation_result_dict)

    return subset_evaluation_result_dict, performance_evaluation_result_dict


def evaluate_performance(
    data_df: pd.DataFrame, cross_validation_indices_list: list, subset_evaluation_result_dict: dict, k: int
) -> dict:
    """
    Evaluate the performance of the feature selection methods.

    This function calculates the performance of the feature selection methods based on the cross-validation indices
    and the subset evaluation results. It then updates the subset evaluation results with the performance results.

    Args:
        data_df: The data to evaluate.
        cross_validation_indices_list: A list of cross-validation indices.
        subset_evaluation_result_dict: A dictionary containing the evaluation results for each feature selection method.
        k: The number of neighbors to use. Defaults to 5.

    Returns:
        A dictionary containing the evaluation results for each feature selection method.
    """
    # check the correct type of the cross-validation indices list
    assert isinstance(cross_validation_indices_list, list), type(cross_validation_indices_list)

    prediction_result_dict = train_model_and_predict_y(
        cross_validation_indices_list, data_df, subset_evaluation_result_dict, k
    )
    performance_evaluation_result_dict = {}
    # calculate performance metrics
    for feature_selection_method in subset_evaluation_result_dict["importance_ranking"].keys():
        performance_evaluation_result_dict[feature_selection_method] = calculate_performance_metrics(
            prediction_result_dict[f"{feature_selection_method}_true"],
            prediction_result_dict[f"{feature_selection_method}_predict"],
            prediction_result_dict[f"{feature_selection_method}_predict_proba"],
        )
    return performance_evaluation_result_dict


def plot_performance(performance_evaluation_result_dict: dict, subset_evaluation_result_dict: dict):
    """
    Plot the performance metrics for each feature selection method.

    This function plots the performance metrics for each feature selection method with plotly.

    Args:
        performance_evaluation_result_dict: A dictionary containing the performance metrics
            for each feature selection method.
        subset_evaluation_result_dict: A dictionary containing the evaluation results for each feature selection method.
    """
    # plot performance metrics for each feature selection method with plotly
    # iterate over all performance metrics
    for performance_metric in performance_evaluation_result_dict["reverse_random_forest_ttest_ind"].keys():
        print(performance_metric)
        if performance_metric in ["fpr", "tpr", "thresholds", "recall", "precision", "auprc"]:
            continue

        # Extract the performance metric, stability, and number of selected features
        data = []
        # methods = [
        #     "standard_random_forest",
        #     "reverse_fraction_p_values_tt",
        #     "reverse_fraction_p_values_mwu",
        # ]  # performance_evaluation_result_dict.keys()
        for method in performance_evaluation_result_dict.keys():
            performance = performance_evaluation_result_dict[method][performance_metric]
            importance_matrix = subset_evaluation_result_dict["importance_ranking"][method]
            stability = stability_estimator.get_stability(importance_matrix)
            subset_sizes = np.sum(np.where(importance_matrix > 0, 1, 0), axis=1)
            num_features = np.mean(subset_sizes, axis=0)
            if "standard" in method:
                method_name = "Standard RF"
            else:
                method_name = method
            # elif "reverse" in method:
            #     method_name = "Reverse RF"
            # else:
            #     raise ValueError("Unknown method.")
            data.append([method_name, performance, stability, num_features])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=["method", "performance", "stability", "num_features"])

        # plot results with plotly
        fig = px.scatter(
            df,
            x="stability",
            y="performance",
            size="num_features",
            color="num_features",
            hover_data=["method"],
            text="method",  # Add method name to each data point
            template="plotly_white",
            # color_continuous_scale=px.colors.qualitative.Pastel2,
            color_continuous_scale=px.colors.qualitative.Set2,
            # color_continuous_scale='Viridis_r'  # Reverse the color scale
            # color_continuous_scale='Greys'  # Use a grey color scale
        )
        fig.update_layout(
            title=f"{data_name}: {performance_metric} vs Stability (colored and shaped by Number of Selected Features)",
            xaxis_title="Stability of Feature Selection",
            yaxis_title=performance_metric,
            legend_title="Number of Selected Features",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        if performance_metric == "auc":
            # save figure as pdf
            pio.write_image(fig, f"{performance_metric}_performance.pdf")
        fig.show()


def calculate_performance_metrics(y_true, y_predict, y_predict_proba):
    """
    Calculate the performance metrics for the predicted labels.

    Args:
        y_true: The true labels.
        y_predict: The predicted labels.
        y_predict_proba: The predicted probabilities.

    Returns:
        A dictionary containing the performance metrics.
    """
    # calculate performance metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    precision, recall, thresholds = precision_recall_curve(y_true, y_predict)

    # brier score loss
    # extract the probability for the positive class
    probability_positive_class = np.array([proba[1] for proba in y_predict_proba])

    performance_dict = {
        "accuracy": accuracy_score(y_true, y_predict),
        "precision": precision_score(y_true, y_predict),
        "recall": recall_score(y_true, y_predict),
        "f1": f1_score(y_true, y_predict),
        "auc": auc(fpr, tpr),
        "log_loss": log_loss(y_true, y_predict_proba),
        "brier_score_loss": brier_score_loss(y_true, probability_positive_class),
        "auprc": average_precision_score(y_true, y_predict),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
    return performance_dict


def train_model_and_predict_y(cross_validation_indices_list, data_df, subset_evaluation_result_dict, k):
    """
    Trains a model and predicts the labels for the test data.

    Args:
        cross_validation_indices_list: A list of cross-validation indices.
        data_df: The data to evaluate.
        subset_evaluation_result_dict: A dictionary containing the feature weights for each feature selection method.
        k: The number of neighbors to use. Defaults to 5.

    Returns:
        A dictionary containing the prediction results for each feature selection method.
    """
    prediction_result_dict = {}
    # iterate over cross-validation indices
    for cv_iteration, (train_indices, test_indices) in enumerate(cross_validation_indices_list):
        # extract training and test data samples
        train_data_df = data_df.iloc[train_indices, :]
        test_data_df = data_df.iloc[test_indices, :]

        # extract x and y values
        x_train = train_data_df.iloc[:, 1:].values
        y_train = train_data_df.iloc[:, 0].values
        x_test = test_data_df.iloc[:, 1:].values
        y_test = test_data_df.iloc[:, 0].values

        # iterate over feature selection methods
        for feature_selection_method in subset_evaluation_result_dict["importance_ranking"].keys():
            # extract weights for feature subset
            feature_subset_weights = subset_evaluation_result_dict["importance_ranking"][feature_selection_method][
                cv_iteration
            ]
            assert (
                feature_subset_weights.size == train_data_df.shape[1] - 1
            ), feature_subset_weights.size  # -1 for label column
            assert np.count_nonzero(feature_subset_weights) > 0, np.count_nonzero(feature_subset_weights)

            # select relevant feature subset
            # extract indices of selected features where feature_subset_weights is nonzero
            relevant_feature_subset_indices = np.flatnonzero(feature_subset_weights)
            feature_subset_x_train = x_train[:, relevant_feature_subset_indices]
            assert feature_subset_x_train.shape == (
                train_data_df.shape[0],
                np.count_nonzero(feature_subset_weights),
            ), feature_subset_x_train.shape
            feature_subset_x_test = x_test[:, relevant_feature_subset_indices]
            assert feature_subset_x_test.shape == (
                test_data_df.shape[0],
                np.count_nonzero(feature_subset_weights),
            ), feature_subset_x_test.shape

            # scale data: choose robust scaler due to the presence of outliers
            robust_scaler = RobustScaler()
            preprocessed_x_train = robust_scaler.fit_transform(feature_subset_x_train)
            preprocessed_x_test = robust_scaler.transform(feature_subset_x_test)

            # train and evaluate the model
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            # knn = KNeighborsClassifier(
            #     n_neighbors=k, metric=WeightedManhattanDistance(feature_subset_weights), n_jobs=-1
            # )
            knn.fit(preprocessed_x_train, y_train)
            predicted_proba_y = knn.predict_proba(preprocessed_x_test)[0]
            predicted_y = knn.predict(preprocessed_x_test)[0]

            # initialize list in feature_selection_method_performance_evaluation_dict
            # for current feature_selection_method
            if f"{feature_selection_method}_predict" not in prediction_result_dict.keys():
                prediction_result_dict[f"{feature_selection_method}_predict"] = []
            if f"{feature_selection_method}_predict_proba" not in prediction_result_dict.keys():
                prediction_result_dict[f"{feature_selection_method}_predict_proba"] = []
            if f"{feature_selection_method}_true" not in prediction_result_dict.keys():
                prediction_result_dict[f"{feature_selection_method}_true"] = []

            # append score to list in feature_selection_method_performance_evaluation_dict
            prediction_result_dict[f"{feature_selection_method}_predict"].append(predicted_y)
            prediction_result_dict[f"{feature_selection_method}_predict_proba"].append(predicted_proba_y)
            prediction_result_dict[f"{feature_selection_method}_true"].append(y_test[0])
    return prediction_result_dict


def evaluate_feature_subsets(
    raw_result_dict: dict,
) -> dict:
    """
    Evaluate feature subsets for different feature selection methods.

    This function extracts for each provided feature selection method the raw feature selection results,
    and analyzes the feature importance matrix.If the feature selection method is "reverse", it applies a
    p-metric_value to select feature subsets for reverse feature selection. It calculates the significant difference
    of reverse feature selection results with p-values based on t-test and Mann-Whitney-U test. It then analyzes the
    feature importance matrix.

    Args:
        raw_result_dict: The raw results dictionary to evaluate.

    Returns:
        A dictionary containing the evaluation results for each feature selection method.
    """
    result_dict = {}
    for key in raw_result_dict.keys():
        # select feature subsets for reverse feature selection
        if key.startswith("reverse"):
            # calculate significant difference of reverse feature selection results
            # based on statistical test
            # statistical_tests = ["ttest_ind", "mannwhitneyu", "welsh", "fraction"]
            statistical_tests = ["ttest_ind", "mannwhitneyu"]
            for test in statistical_tests:
                method_name = f"{key}_{test}"
                print(f"Calculating feature subsets for {method_name}")
                feature_importance_matrix = extract_feature_importance_matrix(raw_result_dict[key], method_name)
                analyze_feature_importance_matrix(
                    result_dict,
                    method_name,
                    feature_importance_matrix,
                )
        # select feature subsets for standard feature selection
        elif key.startswith("standard"):
            feature_importance_matrix = extract_feature_importance_matrix(raw_result_dict[key], key)
            analyze_feature_importance_matrix(
                result_dict,
                key,
                feature_importance_matrix,
            )
        else:
            continue
    # check if the result dictionary is not empty
    assert bool(result_dict)
    return result_dict


def apply_p_value_as_feature_importance(
    result_dict: dict,
    feature_selection_method: str,
    feature_selection_result_cv_list: list,
    p_value: str,
):
    """
    Apply the negative log of the p-value as feature importance.

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
    log_p_value_importance_list = []
    log_p_value_and_fraction_importance_list = []
    # iterate over cross-validation results
    for cv_iteration_result_df in feature_selection_result_cv_list:
        # extract fraction
        fraction_array = cv_iteration_result_df[f"percent_{p_value}"].values

        # extract p-values
        p_values_array = cv_iteration_result_df[p_value].values

        # check if any p-values are zero
        assert not np.any(p_values_array == 0), p_values_array
        # replace nan values with zero
        p_values_array = np.nan_to_num(p_values_array)
        assert not np.isnan(p_values_array).any(), p_values_array
        # check if number of zero values equals number of nan values
        assert np.sum(p_values_array == 0) == np.sum(np.isnan(cv_iteration_result_df[p_value].values)), (
            np.sum(p_values_array == 0),
            np.sum(np.isnan(cv_iteration_result_df[p_value].values)),
        )
        # count the number of p-values between zero and 0.05
        number_of_significant_p_values = np.sum((p_values_array > 0) & (p_values_array < 0.05))
        number_of_not_significant_p_values = np.sum(p_values_array > 0.05)
        assert number_of_significant_p_values + number_of_not_significant_p_values + np.sum(p_values_array == 0) == len(
            p_values_array
        ), (
            number_of_significant_p_values,
            np.sum(p_values_array == 0),
            np.sum(p_values_array > 0.05),
            len(p_values_array),
        )
        # set the p-values greater than 0.05 to zero
        p_values_array[p_values_array > 0.05] = 0
        # check if all p-values are between 0 and 0.05
        assert np.all((p_values_array >= 0) & (p_values_array <= 0.05)), p_values_array
        # Check the number of not NaN values
        assert number_of_significant_p_values == np.count_nonzero(p_values_array)
        p_value_importance_list.append(p_values_array)

        # calculate negative log of the nonzero p-values
        log_p_values_array = np.copy(p_values_array)
        log_p_values_array[p_values_array > 0] = -np.log(p_values_array[p_values_array > 0])
        assert np.all(log_p_values_array >= 0), log_p_values_array
        assert math.isclose(np.min(log_p_values_array), 0), log_p_values_array
        log_p_value_importance_list.append(log_p_values_array)

        # combine negative log p-values and fraction
        log_p_values_fraction_array = np.copy(log_p_values_array)
        # add fraction where log p-value is not zero
        log_p_values_fraction_array[log_p_values_array > 0] = (
            fraction_array[log_p_values_array > 0] + log_p_values_array[log_p_values_array > 0]
        )
        assert np.all(log_p_values_fraction_array >= 0), log_p_values_fraction_array
        assert math.isclose(np.min(log_p_values_fraction_array), 0), log_p_values_fraction_array
        log_p_value_and_fraction_importance_list.append(log_p_values_fraction_array)

        # check number of zeros in all p_values arrays
        assert (
            np.sum(p_values_array == 0)
            == np.sum(log_p_values_array == 0)
            == np.sum(log_p_values_fraction_array == 0)
            == np.sum(np.isnan(cv_iteration_result_df[p_value].values)) + number_of_not_significant_p_values
        ), (
            np.sum(log_p_values_array == 0),
            np.sum(log_p_values_fraction_array == 0),
            np.sum(p_values_array == 0),
            np.sum(np.isnan(cv_iteration_result_df[p_value].values)),
            number_of_not_significant_p_values,
        )

    # convert list of arrays to matrix
    p_value_importance_matrix = np.vstack(p_value_importance_list)
    log_p_value_importance_matrix = np.vstack(log_p_value_importance_list)
    log_p_value_and_fraction_importance_matrix = np.vstack(log_p_value_and_fraction_importance_list)

    # check the shape of the feature importance matrix
    assert (
        p_value_importance_matrix.shape
        == log_p_value_importance_matrix.shape
        == log_p_value_and_fraction_importance_matrix.shape
        == (
            len(feature_selection_result_cv_list),
            feature_selection_result_cv_list[0].shape[0],
        )
    ), p_value_importance_matrix.shape

    # check if the matrices are different
    assert not np.all(p_value_importance_matrix == log_p_value_importance_matrix), (
        p_value_importance_matrix,
        log_p_value_importance_matrix,
    )
    assert not np.all(log_p_value_importance_matrix == log_p_value_and_fraction_importance_matrix), (
        log_p_value_importance_matrix,
        log_p_value_and_fraction_importance_matrix,
    )

    # update result dictionary with p_value feature importance matrix
    result_dict["importance_ranking"][f"{feature_selection_method}_only_raw_{p_value}"] = p_value_importance_matrix
    result_dict["importance_ranking"][f"{feature_selection_method}_neg_log_{p_value}"] = log_p_value_importance_matrix
    result_dict["importance_ranking"][
        f"{feature_selection_method}_neg_log_{p_value}_and_fraction"
    ] = log_p_value_and_fraction_importance_matrix

    # calculate stability of feature selection
    update_result_dictionary(
        result_dict,
        f"{feature_selection_method}_only_raw_{p_value}",
        "stability",
        stability_estimator.get_stability(p_value_importance_matrix),
    )
    update_result_dictionary(
        result_dict,
        f"{feature_selection_method}_neg_log_{p_value}",
        "stability",
        stability_estimator.get_stability(log_p_value_importance_matrix),
    )
    update_result_dictionary(
        result_dict,
        f"{feature_selection_method}_neg_log_{p_value}_and_fraction",
        "stability",
        stability_estimator.get_stability(log_p_value_and_fraction_importance_matrix),
    )

    assert (
        stability_estimator.get_stability(log_p_value_and_fraction_importance_matrix)
        == stability_estimator.get_stability(log_p_value_importance_matrix)
        == stability_estimator.get_stability(p_value_importance_matrix)
    )


def update_result_dictionary(result_dict: dict, feature_selection_method: str, metric_name: str, metric_value: Any):
    """
    Update the result dictionary with a given metric_name-value pair.

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


def scale_feature_importance_matrix(feature_importance_matrix: np.ndarray) -> np.ndarray:
    """
    Scale the feature importance matrix to have all columns sum up to one or zero.

    Args:
        feature_importance_matrix: The feature importance matrix to scale.

    Returns:
        The scaled feature importance matrix.
    """
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


def extract_feature_importance_matrix(
    feature_selection_result_cv_list: List[pd.DataFrame],
    method: str,
) -> np.ndarray:
    """
    Extract the feature importance matrix from the cross-validation results.

    Args:
        feature_selection_result_cv_list: A list of cross-validation results.
        method: The feature selection method used.

    Returns:
        A tuple containing the feature importance matrix and the method name.
    """
    # extract feature selection matrix
    feature_importance_matrix = np.zeros(
        (len(feature_selection_result_cv_list), feature_selection_result_cv_list[0].shape[0])
    )
    # iterate over cross-validation results
    for i, cv_iteration_result_df in enumerate(feature_selection_result_cv_list):
        feature_importance_matrix[i, :] = get_selected_feature_subset(cv_iteration_result_df, method)

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


def get_selected_feature_subset(cv_fold_result_df: pd.DataFrame, method: str) -> np.ndarray:
    """
    Extract the selected feature subset from the cross-validation fold result dataframe.

    Args:
        cv_fold_result_df: The cross-validation fold result dataframe.
        method: The method used for feature selection.

    Returns:
        The selected feature subset.
    """
    if method.startswith("standard"):
        feature_subset_series = cv_fold_result_df["standard_rf"].values
        # sum of all values in the feature subset series must be one
        assert np.isclose(feature_subset_series.sum(), 1.0), feature_subset_series.sum()
    elif method.startswith("standard_shap"):
        feature_subset_series = cv_fold_result_df["shap_values_rf"].values
        # TODO: sum of all values in the feature subset series must be one?
    elif "reverse" in method:
        feature_subset_series = get_feature_subset_for_reverse_feature_selection(cv_fold_result_df, method)
    else:
        raise ValueError("Unknown feature_selection_method.")
    assert feature_subset_series.sum() > 0, feature_subset_series.sum()

    return feature_subset_series


def get_feature_subset_for_reverse_feature_selection(cv_fold_result_df: pd.DataFrame, method_name: str) -> np.ndarray:
    """
    Extract the feature subset for reverse feature selection.

    Args:
        cv_fold_result_df: The cross-validation fold result dataframe.
        method_name: The method name used for feature selection.

    Returns:
        The selected feature subset for reverse feature selection.
    """
    array_of_labeled_oob_error_lists = cv_fold_result_df["labeled"].values
    array_of_unlabeled_oob_error_lists = cv_fold_result_df["unlabeled"].values

    # initialize selected feature subset with zeros as no feature is selected
    selected_feature_subset = np.zeros_like(cv_fold_result_df["labeled"].values)

    # iterate over all error distributions
    for i, (labeled_error_distribution, unlabeled_error_distribution) in enumerate(
        zip(array_of_labeled_oob_error_lists, array_of_unlabeled_oob_error_lists)
    ):
        if labeled_error_distribution is None:
            continue

        # calculate the p-value for the two distributions
        if "ttest_ind" in method_name:
            p_value = ttest_ind(labeled_error_distribution, unlabeled_error_distribution, alternative="less").pvalue
            # p_value2 = ttest_ind(labeled_error_distribution, unlabeled_error_distribution, equal_var=False, alternative="less").pvalue
            #
            # if math.isclose(p_value, p_value2):
            #     if p_value < p_value2:
            #         print("tt", p_value2 - p_value)
            #     else:
            #         print("welsh", p_value - p_value2)
            #
            #     print(p_value, p_value2)

        elif "mannwhitneyu" in method_name:
            p_value = mannwhitneyu(labeled_error_distribution, unlabeled_error_distribution, alternative="less").pvalue
        elif "welsh" in method_name:
            p_value = ttest_ind(
                labeled_error_distribution, unlabeled_error_distribution, equal_var=False, alternative="less"
            ).pvalue
        elif "fraction" in method_name:
            # to not apply the p-value to select the feature subset for reverse feature selection
            p_value = -np.inf
        else:
            raise ValueError(f"Unknown method_name: {method_name}")

        # apply p_value significance level of 0.05 to select feature subset for reverse feature selection
        if p_value < 0.05 and np.mean(unlabeled_error_distribution) > np.mean(labeled_error_distribution):
            # calculate the fraction difference of the two means of the distributions
            fraction = (np.mean(unlabeled_error_distribution) - np.mean(labeled_error_distribution)) / np.mean(
                unlabeled_error_distribution
            )
            # select feature
            selected_feature_subset[i] = fraction
    return selected_feature_subset

    #
    # # replace infinite values with zero
    # cv_fold_result_df["labeled"] = cv_fold_result_df["labeled"].replace([np.inf, -np.inf], 0)
    # cv_fold_result_df["unlabeled"] = cv_fold_result_df["unlabeled"].replace([np.inf, -np.inf], 0)
    #
    # assert not cv_fold_result_df["labeled"].isnull().values.any(), cv_fold_result_df["labeled"].isnull().sum()
    # assert not cv_fold_result_df["unlabeled"].isnull().values.any(), cv_fold_result_df["unlabeled"].isnull().sum()
    #
    # # extract feature subset for reverse feature selection based on the percentage difference
    # # between labeled and unlabeled OOB error
    # selected_feature_subset = (cv_fold_result_df["unlabeled"] - cv_fold_result_df["labeled"]) / cv_fold_result_df[
    #     "unlabeled"
    # ]
    # # replace nan values with zero
    # selected_feature_subset = selected_feature_subset.fillna(0)
    # # check if the feature subset contains nan values
    # assert not selected_feature_subset.isnull().values.any(), selected_feature_subset.isnull().sum()
    #
    # # check if the feature subset contains infinite values
    # assert not np.isinf(selected_feature_subset).values.any(), selected_feature_subset.isinf().sum()
    #
    # # check if the feature subset contains negative values
    # assert not (selected_feature_subset < 0).values.any(), selected_feature_subset[selected_feature_subset < 0]
    #
    # # if "percent" not in cv_fold_result_df.columns:
    # cv_fold_result_df.insert(2, f"percent_{p_value}", selected_feature_subset, allow_duplicates=False)
    #
    # if p_value is not None:
    #     # apply p_value significance level of 0.05 to select feature subset for reverse feature selection
    #     for feature in selected_feature_subset.index:
    #         if cv_fold_result_df[p_value][feature] >= 0.05:
    #             selected_feature_subset[feature] = 0
    #
    # return selected_feature_subset, method_name


# parse settings from toml
# with open("./settings.toml", "r") as file:
#     meta_data = toml.load(file)
# # data_name = "artificial221"
# # input_data_df = pd.read_csv(meta_data["data"]["path"]).iloc[:, :221]
# data_name = "colon00"
# # # data_name = "prostate"
# input_data_df = load_data_with_standardized_sample_size("colon")
# # TODO rest of the data as test data
# subset_result_dict_final, performance_result_dict_final = evaluate(
#     input_data_df,
#     f"/home/sigrun/PycharmProjects/reverse_feature_selection/results/{data_name}_result_dict.pkl",
#     feature_selection_methods=["standard", "standard_shap", "reverse"],
# )
# print(performance_result_dict_final)


def evaluate_reproducibility(data_df: pd.DataFrame, base_path: str, repeated_experiments: list[str]):
    """
    Evaluate the reproducibility of the feature selection results.

    This function evaluates the reproducibility of the feature selection results for each provided feature selection method.

    Args:
        data_df: The underlying original data for the given experiments.
        base_path: The base path to the pickled results files.
        repeated_experiments: A list of the experiment ids of repeated experiments to evaluate.

    Returns:
        A dictionary containing the evaluation results for each feature selection method.
    """
    results_dict = {}

    for experiment_id in repeated_experiments:
        subset_result_dict_final, performance_result_dict_final = evaluate(
            data_df,
            f"{base_path}/{experiment_id}_result_dict.pkl",
        )
        for selection_method in subset_result_dict_final["stability"].keys():
            if f"{selection_method}_stability" not in results_dict.keys():
                results_dict[f"{selection_method}_stability"] = []
            results_dict[f"{selection_method}_stability"].append(
                subset_result_dict_final["stability"][selection_method]
            )

            if f"{selection_method}_mean_subset_size" not in results_dict.keys():
                results_dict[f"{selection_method}_mean_subset_size"] = []
            results_dict[f"{selection_method}_mean_subset_size"].append(
                subset_result_dict_final["mean_subset_size"][selection_method]
            )

        for selection_method in performance_result_dict_final.keys():
            if f"{selection_method}_auc" not in results_dict.keys():
                results_dict[f"{selection_method}_auc"] = []
            results_dict[f"{selection_method}_auc"].append(performance_result_dict_final[selection_method]["auc"])

            if f"{selection_method}_f1" not in results_dict.keys():
                results_dict[f"{selection_method}_f1"] = []
            results_dict[f"{selection_method}_f1"].append(performance_result_dict_final[selection_method]["f1"])

            if f"{selection_method}_log_loss" not in results_dict.keys():
                results_dict[f"{selection_method}_log_loss"] = []
            results_dict[f"{selection_method}_log_loss"].append(
                performance_result_dict_final[selection_method]["log_loss"]
            )

            if f"{selection_method}_brier_score_loss" not in results_dict.keys():
                results_dict[f"{selection_method}_brier_score_loss"] = []
            results_dict[f"{selection_method}_brier_score_loss"].append(
                performance_result_dict_final[selection_method]["brier_score_loss"]
            )
        # compare the results
        print(subset_result_dict_final)


list_of_experiments = ["colon00", "colon_s10"]
pickle_base_path = f"results"
data_name = "Colon Cancer"
input_data_df = load_data_with_standardized_sample_size("colon")

evaluate_reproducibility(input_data_df, pickle_base_path, list_of_experiments)
