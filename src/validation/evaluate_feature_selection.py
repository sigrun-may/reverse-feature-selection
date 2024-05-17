# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Evaluation of feature selection results."""

import math
import os
import pickle
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier

import stability_estimator
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

from src.reverse_feature_selection.data_loader_tools import (
    load_data_with_standardized_sample_size,
    load_train_test_data_for_standardized_sample_size,
)
from src.validation.weighted_manhattan_distance import WeightedManhattanDistance


def evaluate(
        data_df: pd.DataFrame,
        data_df_test: pd.DataFrame,
        raw_result_dict: dict,
) -> Tuple[dict, dict, dict]:
    """
    Evaluate the feature selection results.

    This function loads the pickled results from the specified path, extracts the feature selection results,
    and evaluates them for each provided feature selection method.

    Args:
        data_df: The data to evaluate.
        data_df_test: The test data to evaluate.
        raw_result_dict: The raw results dictionary to evaluate.

    Returns:
        A dictionary containing the evaluation results for each feature selection method.
    """
    # evaluate feature selection results
    subset_evaluation_result_dict = evaluate_feature_subsets(raw_result_dict)

    # evaluate performance
    performance_evaluation_result_dict = evaluate_cv_performance(
        data_df, raw_result_dict["indices"], subset_evaluation_result_dict, k=5
    )

    # evaluate test performance
    test_performance_evaluation_result_dict = evaluate_test_performance(
        data_df, data_df_test, subset_evaluation_result_dict, k=5
    )
    return subset_evaluation_result_dict, performance_evaluation_result_dict, test_performance_evaluation_result_dict


def evaluate_test_performance(train_data, test_data, subset_evaluation_result_dict, k):
    performance_result_dict = {}
    # calculate performance metrics
    for feature_selection_method in subset_evaluation_result_dict["importance_ranking"].keys():
        feature_importances = subset_evaluation_result_dict["importance_ranking"][feature_selection_method]
        # sum up all feature importances
        feature_importance = np.sum(feature_importances, axis=0)
        y_predict, y_predict_proba, y_true = train_model_and_predict_y(train_data, test_data, feature_importance, k)
        performance_result_dict[feature_selection_method] = calculate_performance_metrics(
            y_true, y_predict, y_predict_proba
        )
    return performance_result_dict


def evaluate_cv_performance(
        data_df: pd.DataFrame, cross_validation_indices_list: list, feature_importances, k: int
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
    # calculate performance metrics
    y_predict, y_predict_proba, y_true = validate_cv(cross_validation_indices_list, data_df, feature_importances, k)
    return calculate_performance_metrics(y_true, y_predict, y_predict_proba)


def plot_performance(
        performance_evaluation_result_dict: dict, subset_evaluation_result_dict: dict, feature_selection_methods: list
):
    """
    Plot the performance metrics for each feature selection method.

    This function plots the performance metrics for each feature selection method with plotly.

    Args:
        performance_evaluation_result_dict: A dictionary containing the performance metrics
            for each feature selection method.
        subset_evaluation_result_dict: A dictionary containing the evaluation results for each feature selection method.
        feature_selection_methods: A list of feature selection methods to plot.
    """
    # plot performance metrics for each feature selection method with plotly
    # iterate over all performance metrics
    for performance_metric in performance_evaluation_result_dict["reverse_random_forest_ttest_ind"].keys():
        print(performance_metric)
        if performance_metric in ["fpr", "tpr", "thresholds", "recall", "precision", "auprc"]:
            continue

        # Extract the performance metric, stability, and number of selected features
        data = []
        for method in performance_evaluation_result_dict.keys():
            if feature_selection_methods is not None and method not in feature_selection_methods:
                print(f"Skipping {method}")
                continue
            performance = performance_evaluation_result_dict[method][performance_metric]
            importance_matrix = subset_evaluation_result_dict["importance_ranking"][method]
            stability = stability_estimator.get_stability(importance_matrix)
            subset_sizes = np.sum(np.where(importance_matrix > 0, 1, 0), axis=1)
            num_features = np.mean(subset_sizes, axis=0)

            # TODO set method name for plot
            # if "standard" in method:
            #     method_name = "Standard RF"
            # else:
            #     method_name = method
            # elif "reverse" in method:
            #     method_name = "Reverse RF"
            # else:
            #     raise ValueError("Unknown method.")
            method_name = method
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
        "Accuracy": accuracy_score(y_true, y_predict),
        "precision": precision_score(y_true, y_predict),
        "recall": recall_score(y_true, y_predict),
        "F1": f1_score(y_true, y_predict),
        "AUC": auc(fpr, tpr),
        "log loss": log_loss(y_true, y_predict_proba),
        "Brier Score": brier_score_loss(y_true, probability_positive_class),
        "auprc": average_precision_score(y_true, y_predict),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
    return performance_dict


def validate_cv(cross_validation_indices_list: list, data_df: pd.DataFrame, feature_importances, k: int):
    y_predict = []
    y_predict_proba = []
    y_true = []
    # iterate over cross-validation indices
    for cv_iteration, (train_indices, test_indices) in enumerate(cross_validation_indices_list):
        feature_importance = feature_importances[cv_iteration]
        # extract training and test data samples
        train_data = data_df.iloc[train_indices, :]
        test_data = data_df.iloc[test_indices, :]
        y_pred, y_pred_proba, true_y = train_model_and_predict_y(train_data, test_data, feature_importance, k)
        y_predict.append(y_pred[0])
        y_predict_proba.append(y_pred_proba[0])
        y_true.append(true_y[0])
    return y_predict, y_predict_proba, y_true


def train_model_and_predict_y(train_data, test_data, feature_importance, k):
    """
    Trains a model and predicts the labels for the test data.

    Args:
        train_data: The training data.
        test_data: The test data.
        feature_importance: The feature importance.
        k: The number of neighbors to use. Defaults to 5.

    Returns:
        A dictionary containing the prediction results for each feature selection method.
    """
    # extract x and y values
    x_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    feature_subset_weights = feature_importance
    assert feature_subset_weights.size == train_data.shape[1] - 1, feature_subset_weights.size  # -1 for label column
    assert np.count_nonzero(feature_subset_weights) > 0, np.count_nonzero(feature_subset_weights)

    # select relevant feature subset
    # extract indices of selected features where feature_subset_weights is nonzero
    relevant_feature_subset_indices = np.flatnonzero(feature_subset_weights)
    feature_subset_x_train = x_train[:, relevant_feature_subset_indices]
    assert feature_subset_x_train.shape == (
        train_data.shape[0],
        np.count_nonzero(feature_subset_weights),
    ), feature_subset_x_train.shape
    feature_subset_x_test = x_test[:, relevant_feature_subset_indices]
    assert feature_subset_x_test.shape == (
        test_data.shape[0],
        np.count_nonzero(feature_subset_weights),
    ), feature_subset_x_test.shape

    # scale data: choose robust scaler due to the presence of outliers
    robust_scaler = RobustScaler()
    preprocessed_x_train = robust_scaler.fit_transform(feature_subset_x_train)
    preprocessed_x_test = robust_scaler.transform(feature_subset_x_test)

    # train and evaluate the model
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    # knn = RandomForestClassifier()
    # knn = KNeighborsClassifier(
    #     n_neighbors=k, metric=WeightedManhattanDistance(feature_subset_weights[relevant_feature_subset_indices]),
    #     n_jobs=1
    # )
    knn.fit(preprocessed_x_train, y_train)
    predicted_proba_y = knn.predict_proba(preprocessed_x_test)
    predicted_y = knn.predict(preprocessed_x_test)
    return predicted_y, predicted_proba_y, y_test


def train_model_and_predict_y2(cross_validation_indices_list, data_df, subset_evaluation_result_dict, k):
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
            # knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric=WeightedManhattanDistance(feature_subset_weights[relevant_feature_subset_indices]),
                n_jobs=1,
            )
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


def evaluate_feature_subsets(raw_result_dict: dict) -> dict:
    """
    Evaluate feature subsets for different feature selection methods.

    This function extracts for each provided feature selection method the raw feature selection results,
    and analyzes the feature importance matrix.If the feature selection method is "reverse", it applies a
    p-metric_value to select feature subsets for reverse feature selection. It calculates the significant difference
    of reverse feature selection results with p-values based on Mann-Whitney-U test. It then analyzes the
    feature importance matrix.

    Args:
        raw_result_dict: The raw results dictionary to evaluate.

    Returns:
        A dictionary containing the evaluation results for each feature selection method.
    """
    methods_result_dict = {}
    for key in raw_result_dict.keys():
        # select feature subsets for reverse feature selection
        if key.startswith("reverse"):
            # calculate significant difference of reverse feature selection results based on Mann-Whitney-U test
            feature_importance_matrix = extract_feature_importance_matrix(raw_result_dict[key], key)
            analyze_feature_importance_matrix(
                methods_result_dict,
                key,
                feature_importance_matrix,
            )
        # select feature subsets for standard feature selection
        elif key.startswith("standard"):
            feature_importance_matrix = extract_feature_importance_matrix(raw_result_dict[key], key)
            analyze_feature_importance_matrix(
                methods_result_dict,
                key,
                feature_importance_matrix,
            )
        else:
            continue
    # check if the result dictionary is not empty
    assert bool(methods_result_dict)
    return methods_result_dict


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


def analyze_feature_importance_matrix(feature_importance_matrix: np.ndarray) -> dict:
    """
    Analyzes a feature importance matrix and updates a result dictionary with various metrics.

    This function calculates the stability, robustness, subset size, mean subset size, importance ranking,
    and feature weights based on the feature importance matrix. It then returns the result dictionary
    with these metrics.

    Args:
        feature_importance_matrix: The feature importance matrix to analyze.

    Returns:
        Analysis results for the feature importance matrix.
    """
    result_dict = {
        # calculate stability of feature selection
        "stability": stability_estimator.get_stability(feature_importance_matrix),
        # calculate robustness of feature selection
        "robustness_array": np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=0),
        # calculate subset size of feature selection
        "subset_size_array": np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=1),
        # calculate mean subset size
        "mean_subset_size": np.mean(np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=1), axis=0),
        # calculate importance ranking
        "importance_ranking": feature_importance_matrix,
    }

    return result_dict


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
    # initialize feature importance matrix with zeros
    feature_importance_matrix = np.zeros(
        (len(feature_selection_result_cv_list), len(feature_selection_result_cv_list[0]))
    )
    # iterate over cross-validation results and populate the matrix with feature importances
    for i, cv_iteration_result_df in enumerate(feature_selection_result_cv_list):
        feature_importance_matrix[i, :] = get_selected_feature_subset(cv_iteration_result_df, method)

    # check of each row contains at least one nonzero metric_value
    if not np.any(np.sum(feature_importance_matrix, axis=1) > 0):
        raise ValueError("Each fold must have at least one feature with nonzero importance.")

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

        # calculate the p-value for the two distributions based on the mann-whitney-u test
        p_value = mannwhitneyu(labeled_error_distribution, unlabeled_error_distribution, alternative="less").pvalue
        # if "ttest_ind" in method_name:
        #     p_value = ttest_ind(labeled_error_distribution, unlabeled_error_distribution, alternative="less").pvalue
        #     # p_value2 = ttest_ind(labeled_error_distribution, unlabeled_error_distribution, equal_var=False, alternative="less").pvalue
        #     #
        #     # if math.isclose(p_value, p_value2):
        #     #     if p_value < p_value2:
        #     #         print("tt", p_value2 - p_value)
        #     #     else:
        #     #         print("welsh", p_value - p_value2)
        #     #
        #     #     print(p_value, p_value2)
        # elif "mannwhitneyu" in method_name:
        #     p_value = mannwhitneyu(labeled_error_distribution, unlabeled_error_distribution, alternative="less").pvalue
        # elif "welsh" in method_name:
        #     p_value = ttest_ind(
        #         labeled_error_distribution, unlabeled_error_distribution, equal_var=False, alternative="less"
        #     ).pvalue
        # elif "fraction" in method_name:
        #     # to not apply the p-value to select the feature subset for reverse feature selection
        #     p_value = -np.inf
        # else:
        #     raise ValueError(f"Unknown method_name: {method_name}")

        # apply p_value significance level of 0.05 to select feature subset for reverse feature selection
        if p_value < 0.05:
            # calculate the fraction difference of the two means of the distributions
            fraction = (np.median(unlabeled_error_distribution) - np.median(labeled_error_distribution)) / np.median(
                unlabeled_error_distribution
            )
            # fraction2 = (np.mean(unlabeled_error_distribution) - np.mean(labeled_error_distribution)) / np.mean(
            #     unlabeled_error_distribution
            # )
            # select feature
            selected_feature_subset[i] = fraction

            # # check if the error distributions are normally distributed
            # if normaltest(unlabeled_error_distribution).pvalue < 0.05:
            #     print(normaltest(unlabeled_error_distribution))
            #     print(normaltest(labeled_error_distribution))
            #     print(fraction, fraction2, p_value)
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


def append_to_results_dict(results_dict, method_key, metric, value):
    """
    Appends the value to the list of metrics for the given method in the results dictionary.
    If the method or metric does not exist, it initializes them.
    """
    if method_key not in results_dict:
        results_dict[method_key] = defaultdict(list)
    results_dict[method_key][metric].append(value)


def evaluate_reproducibility(
        data_df: pd.DataFrame, test_data: pd.DataFrame, base_path: str, repeated_experiments: list[str], k: int
) -> dict:
    """
    Evaluate the feature selection results and their reproducibility for each provided feature selection method.

    Args:
        data_df: The underlying original data for the given experiments.
        test_data: The test data to evaluate.
        base_path: The base path to the pickled results files.
        repeated_experiments: A list of the experiment ids of repeated experiments to evaluate.
        k: The number of neighbors to use. Defaults to 5.

    Returns:
        Evaluation results for each feature selection method.
    """
    results_dict = {}
    # iterate over all repeated experiments
    for experiment_id in repeated_experiments:
        print(f"Evaluating experiment {experiment_id}")
        file_path = os.path.join(base_path, f"{experiment_id}_result_dict.pkl")

        try:
            with (open(file_path, "rb") as file):
                raw_result_dict = pickle.load(file)

                # iterate over all feature selection methods
                for key in raw_result_dict.keys():
                    if "reverse" in key or "standard" in key:
                        # select feature subsets
                        feature_importance_matrix = extract_feature_importance_matrix(raw_result_dict[key], key)
                        append_to_results_dict(results_dict, key, "stability", stability_estimator.get_stability(feature_importance_matrix))
                        append_to_results_dict(results_dict, key, "robustness_array", np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=0))
                        append_to_results_dict(results_dict, key, "subset_size_array", np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=1))
                        append_to_results_dict(results_dict, key, "mean_subset_size", np.mean(np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=1), axis=0))
                        append_to_results_dict(results_dict, key, "importance_ranking", feature_importance_matrix)

                        # evaluate loo cv performance
                        y_predict, y_predict_proba, y_true = validate_cv(
                            raw_result_dict["indices"], data_df, feature_importance_matrix, k
                        )
                        append_performance_metrics(key, "cv_performance", results_dict, y_predict, y_predict_proba, y_true)

                        # evaluate test performance
                        # aggregate feature importance matrix
                        # TODO train complete train data without loo cv?
                        y_predict1, y_predict_proba1, y_true1 = train_model_and_predict_y(data_df, test_data,
                                                                                       feature_importance=np.sum(feature_importance_matrix, axis=0), k=k)
                        append_performance_metrics(key, "test_performance", results_dict, y_predict1, y_predict_proba1, y_true1)

            #     # evaluate performance
            #     cv_performance_result_dict = evaluate_cv_performance(
            #         data_df, raw_result_dict["indices"], subset_evaluation_result_dict, k=5
            #     )
            #     # evaluate test performance
            #     test_performance_evaluation_result_dict = evaluate_test_performance(
            #         data_df, test_data, subset_evaluation_result_dict, k=5
            #     )
            # for selection_method in subset_evaluation_result_dict["stability"].keys():
            #     # save cv evaluation results
            #     results_dict_cv[selection_method]["stability"].append(
            #         subset_evaluation_result_dict["stability"][selection_method]
            #     )
            #     results_dict_cv[selection_method]["mean_subset_size"].append(
            #         subset_evaluation_result_dict["mean_subset_size"][selection_method]
            #     )
            #     results_dict_cv[selection_method]["AUC"].append(cv_performance_result_dict[selection_method]["AUC"])
            #     results_dict_cv[selection_method]["Accuracy"].append(
            #         cv_performance_result_dict[selection_method]["Accuracy"]
            #     )
            #     results_dict_cv[selection_method]["F1"].append(cv_performance_result_dict[selection_method]["F1"])
            #     results_dict_cv[selection_method]["Brier Score"].append(
            #         cv_performance_result_dict[selection_method]["Brier Score"]
            #     )
            #
            #     # save test evaluation results
            #     results_dict_test[selection_method]["stability"].append(
            #         subset_evaluation_result_dict["stability"][selection_method]
            #     )
            #     results_dict_test[selection_method]["mean_subset_size"].append(
            #         subset_evaluation_result_dict["mean_subset_size"][selection_method]
            #     )
            #     results_dict_test[selection_method]["AUC"].append(
            #         test_performance_evaluation_result_dict[selection_method]["AUC"]
            #     )
            #     results_dict_test[selection_method]["Accuracy"].append(
            #         test_performance_evaluation_result_dict[selection_method]["Accuracy"]
            #     )
            #     results_dict_test[selection_method]["F1"].append(
            #         test_performance_evaluation_result_dict[selection_method]["F1"]
            #     )
            #     results_dict_test[selection_method]["Brier Score"].append(
            #         test_performance_evaluation_result_dict[selection_method]["Brier Score"]
            #     )
        except FileNotFoundError:
            print(f"File {file_path} not found.")
    return results_dict


def append_performance_metrics(feature_selection_method, performance_evaluation_method, results_dict, y_predict, y_predict_proba, y_true):
    for metric, value in calculate_performance_metrics(y_true, y_predict, y_predict_proba).items():
        # check if cv_performance is already a key in the dictionary
        if performance_evaluation_method not in results_dict[feature_selection_method]:
            results_dict[feature_selection_method][performance_evaluation_method] = {}
        # check if the metric is already a key in the dictionary
        if metric not in results_dict[feature_selection_method][performance_evaluation_method]:
            results_dict[feature_selection_method][performance_evaluation_method][metric] = [value]
        else:
            results_dict[feature_selection_method][performance_evaluation_method][metric].append(value)


def plot_average_results(evaluated_results_dict: dict):
    """
    Plot the average results and the standard deviation for the given results dictionary.

    Args:
        evaluated_results_dict: The dictionary containing the evaluated results to plot.
    """
    for performance_evaluation_method in ["cv_performance", "test_performance"]:
        for performance_metric in ["AUC", "Accuracy", "F1", "Brier Score"]:
            # Create a DataFrame
            df = pd.DataFrame(
                columns=[
                    "method",
                    "performance",
                    "stability",
                    "num_features",
                    "e_performance",
                    "e_stability",
                    "e_num_features",
                ]
            )
            for feature_selection_method in evaluated_results_dict.keys():
                df.loc[len(df)] = [feature_selection_method,
                                   np.mean(evaluated_results_dict[feature_selection_method][performance_evaluation_method][
                                               performance_metric]),
                                   np.mean(evaluated_results_dict[feature_selection_method]["stability"]),
                                   np.mean(evaluated_results_dict[feature_selection_method]["mean_subset_size"]),
                                   np.std(evaluated_results_dict[feature_selection_method][performance_evaluation_method][
                                             performance_metric]),
                                   np.std(evaluated_results_dict[feature_selection_method]["stability"]),
                                   np.std(evaluated_results_dict[feature_selection_method]["mean_subset_size"]),
                                   ]
            fig = px.scatter(
                df,
                x="stability",
                y="performance",
                error_y="e_performance",
                error_x="e_stability",
                size="num_features",
                color="num_features",
                hover_data=["method"],
                text="method",  # Add method name to each data point
                template="plotly_white",
            )
            fig.update_traces(textposition="top left")
            fig.update_xaxes(range=[0, 1.0])
            fig.update_layout(
                title=f"{data_name}: {performance_metric} vs Stability (colored and shaped by number of selected "
                      f"features)",
                xaxis_title="Stability of Feature Selection",
                yaxis_title=f"{performance_metric} - {performance_evaluation_method}",
                legend_title="Average Number of Selected Features",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            #     fig.update_layout(
            #         title=f"{data_name}: {performance_metric} vs Stability (colored and shaped by number of selected features)",
            #         xaxis_title="Stability of Feature Selection",
            #         yaxis_title=performance_metric,
            #         legend_title="Number of Selected Features",
            #         legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            #     )
            fig.show()

    #     # plot mean results with plotly
    #     fig = px.scatter(
    #         df,
    #         x="stability",
    #         y="performance",
    #         error_y="e_performance",
    #         error_x="e_stability",
    #         size="num_features",
    #         color="num_features",
    #         hover_data=["method"],
    #         text="method",  # Add method name to each data point
    #         template="plotly_white",
    #         # color_continuous_scale=px.colors.qualitative.Pastel2,
    #         # color_continuous_scale=px.colors.qualitative.Set2,
    #         # color_continuous_scale='Viridis'
    #         # color_continuous_scale='Greys'  # Use a grey color scale
    #         # color_continuous_scale="smoker",
    #         # color_
    #     )
    #     fig.update_traces(textposition="top left")
    #     fig.update_xaxes(range=[0, 1.0])
    #     fig.update_layout(
    #         title=f"{data_name}: {performance_metric} vs Stability (colored and shaped by number of selected features)",
    #         xaxis_title="Stability of Feature Selection",
    #         yaxis_title=performance_metric,
    #         legend_title="Number of Selected Features",
    #         legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    #     )
    #
    #
    #
    # data_for_df = [
    #     {
    #         "method": method,
    #         "performance": np.mean(evaluated_results_dict[method][performance_metric]),
    #         "stability": np.mean(evaluated_results_dict[method]["stability"]),
    #         "num_features": np.mean(evaluated_results_dict[method]["mean_subset_size"]),
    #         "e_performance": np.std(evaluated_results_dict[method][performance_metric]),
    #         "e_stability": np.std(evaluated_results_dict[method]["stability"]),
    #         "e_num_features": np.std(evaluated_results_dict[method]["mean_subset_size"]),
    #     }
    #     for performance_metric in evaluated_results_dict[method].keys()
    #     for method in evaluated_results_dict.keys()
    # ]
    #
    # df = pd.DataFrame(data_for_df)
    #
    # fig = px.scatter(
    #     df,
    #     x="stability",
    #     y="performance",
    #     error_y="e_performance",
    #     error_x="e_stability",
    #     size="num_features",
    #     color="method",
    #     hover_data=["method"],
    #     template="plotly_white",
    # )
    #
    # fig.update_traces(textposition="top left")
    # fig.update_xaxes(range=[0, 1.0])
    # fig.update_layout(
    #     title=f"Performance vs Stability (colored by method, sized by number of selected features)",
    #     xaxis_title="Stability of Feature Selection",
    #     yaxis_title="Performance",
    #     legend_title="Method",
    #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    # )
    #
    #
    #
    # # iterate over all performance metrics
    # for performance_metric in evaluated_results_dict["reverse_random_forest"].keys():
    #     if performance_metric in ["stability", "mean_subset_size"]:
    #         continue
    #
    #     # Create a DataFrame
    #     df = pd.DataFrame(
    #         columns=[
    #             "method",
    #             "performance",
    #             "stability",
    #             "num_features",
    #             "e_performance",
    #             "e_stability",
    #             "e_num_features",
    #         ]
    #     )
    #
    #     # Extract the performance metric, stability, and number of selected features
    #     for method in evaluated_results_dict.keys():
    #         df.loc[len(df)] = [
    #             method,
    #             np.mean(evaluated_results_dict[method][performance_metric]),
    #             np.mean(evaluated_results_dict[method]["stability"]),
    #             np.mean(evaluated_results_dict[method]["mean_subset_size"]),
    #             # # calculate the standard error of the mean (sem) of performance, stability, and num_features
    #             # # doi: 10.1097/jp9.0000000000000024
    #             # sem(evaluated_results_dict[method][performance_metric]),
    #             # sem(evaluated_results_dict[method]["stability"]),
    #             # sem(evaluated_results_dict[method]["mean_subset_size"]),
    #             # calculate the standard deviation of performance, stability, and num_features
    #             np.std(evaluated_results_dict[method][performance_metric]),
    #             np.std(evaluated_results_dict[method]["stability"]),
    #             np.std(evaluated_results_dict[method]["mean_subset_size"]),
    #             # # calculate the maximum of performance, stability, and num_features
    #             # np.max(evaluated_results_dict[method][performance_metric]),
    #             # np.max(evaluated_results_dict[method]["stability"]),
    #             # np.max(evaluated_results_dict[method]["mean_subset_size"]),
    #             # # calculate the minimum of performance, stability, and num_features
    #             # np.min(evaluated_results_dict[method][performance_metric]),
    #             # np.min(evaluated_results_dict[method]["stability"]),
    #             # np.min(evaluated_results_dict[method]["mean_subset_size"]),
    #         ]
    #
    #     # plot mean results with plotly
    #     fig = px.scatter(
    #         df,
    #         x="stability",
    #         y="performance",
    #         error_y="e_performance",
    #         error_x="e_stability",
    #         size="num_features",
    #         color="num_features",
    #         hover_data=["method"],
    #         text="method",  # Add method name to each data point
    #         template="plotly_white",
    #         # color_continuous_scale=px.colors.qualitative.Pastel2,
    #         # color_continuous_scale=px.colors.qualitative.Set2,
    #         # color_continuous_scale='Viridis'
    #         # color_continuous_scale='Greys'  # Use a grey color scale
    #         # color_continuous_scale="smoker",
    #         # color_
    #     )
    #     fig.update_traces(textposition="top left")
    #     fig.update_xaxes(range=[0, 1.0])
    #     fig.update_layout(
    #         title=f"{data_name}: {performance_metric} vs Stability (colored and shaped by number of selected features)",
    #         xaxis_title="Stability of Feature Selection",
    #         yaxis_title=performance_metric,
    #         legend_title="Number of Selected Features",
    #         legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    #     )
    #     # TODO Save results to pdf
    #     # if performance_metric == "auc":
    #     #     # save figure as pdf
    #     #     pio.write_image(fig, f"{performance_metric}_performance.pdf")
    #     fig.show()
    #
    #     # # check if the list of values for the performance metric contains more than one value
    #     # if len(evaluated_results_dict["standard_random_forest"][performance_metric]) > 1:
    #     #     # plot results of performance as plotly bar chart and add error bars
    #     #     # for standard error of the mean (sem) of performance
    #     #     fig = go.Figure()
    #     #     fig.add_trace(
    #     #         go.Bar(
    #     #             x=df["method"],
    #     #             y=df["performance"],
    #     #             name="Performance",
    #     #             error_y=dict(type="data", array=df["e_performance"]),
    #     #         )
    #     #     )
    #     #     fig.update_layout(
    #     #         title=f"{data_name}: {performance_metric}",
    #     #         xaxis_title="Feature Selection Method",
    #     #         yaxis_title=performance_metric,
    #     #     )
    #     #     fig.show()
    #     #
    #     #     # plot results of stability as plotly bar chart and add error bars
    #     #     # for standard error of the mean (sem) of stability
    #     #     fig = go.Figure()
    #     #     fig.add_trace(
    #     #         go.Bar(
    #     #             x=df["method"],
    #     #             y=df["stability"],
    #     #             name="Stability",
    #     #             error_y=dict(type="data", array=df["e_stability"]),
    #     #         )
    #     #     )
    #     #     fig.update_layout(
    #     #         title=f"{data_name}: Stability",
    #     #         xaxis_title="Feature Selection Method",
    #     #         yaxis_title="Stability",
    #     #     )
    #     #     fig.show()
    #     #
    #     #     # plot results of num_features as plotly bar chart and add error bars
    #     #     # for standard error of the mean (sem) of num_features
    #     #     fig = go.Figure()
    #     #     fig.add_trace(
    #     #         go.Bar(
    #     #             x=df["method"],
    #     #             y=df["num_features"],
    #     #             name="Number of Selected Features",
    #     #             error_y=dict(type="data", array=df["e_num_features"]),
    #     #         )
    #     #     )
    #     #     fig.update_layout(
    #     #         title=f"{data_name}: Number of Selected Features",
    #     #         xaxis_title="Feature Selection Method",
    #     #         yaxis_title="Number of Selected Features",
    #     #     )
    #     #     fig.show()


list_of_experiments = [
    "colon00",
    "colon_s10",
    # "colon_s20",
    # "colon_s30",
    # "colon_s40",
]
# list_of_experiments = ["colon00"]
pickle_base_path = f"results"
data_name = "Colon Cancer"
test_data_df, input_data_df = load_train_test_data_for_standardized_sample_size("colon")

# list_of_experiments = [
#     "prostate_s00",
# ]
# pickle_base_path = f"results"
# data_name = "Prostate Cancer"
# input_data_df = load_data_with_standardized_sample_size("prostate")

result_dict_cv = evaluate_reproducibility(
    input_data_df, test_data_df, pickle_base_path, list_of_experiments, 5
)
plot_average_results(result_dict_cv)
