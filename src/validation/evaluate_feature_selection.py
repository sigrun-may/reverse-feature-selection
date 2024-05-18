# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Evaluation of feature selection results."""

import os
import pickle
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import stability_estimator
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer

from src.reverse_feature_selection.data_loader_tools import \
    load_train_test_data_for_standardized_sample_size


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
    # robust_scaler = RobustScaler()
    # robust_scaler = PowerTransformer()
    robust_scaler = QuantileTransformer(n_quantiles=train_data.shape[0], output_distribution="normal")

    preprocessed_x_train = robust_scaler.fit_transform(feature_subset_x_train)
    preprocessed_x_test = robust_scaler.transform(feature_subset_x_test)

    # train and evaluate the model
    if k is None:
        clf = RandomForestClassifier()
    else:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    clf.fit(preprocessed_x_train, y_train)
    predicted_proba_y = clf.predict_proba(preprocessed_x_test)
    predicted_y = clf.predict(preprocessed_x_test)
    return predicted_y, predicted_proba_y, y_test


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
            # deselect feature
            continue

        # calculate the p-value for the two distributions based on the mann-whitney-u test
        p_value = mannwhitneyu(labeled_error_distribution, unlabeled_error_distribution, alternative="less").pvalue

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
    return selected_feature_subset


def append_to_results_dict(results_dict, method_key, metric, value):
    """
    Appends the value to the list of metrics for the given method in the results dictionary.
    If the method or metric does not exist, it initializes them.
    """
    if method_key not in results_dict:
        results_dict[method_key] = defaultdict(list)
    results_dict[method_key][metric].append(value)


def evaluate(
    data_df: pd.DataFrame,
    test_data: pd.DataFrame,
    pickle_base_path: str,
    repeated_experiments: list[str],
    k: int | None,
) -> dict:
    """
    Evaluate the feature selection results and their reproducibility for each provided feature selection method.

    Args:
        data_df: The underlying original data for the given experiments.
        test_data: The test data to evaluate.
        pickle_base_path: The base path to the pickled results files.
        repeated_experiments: A list of the experiment ids of repeated experiments to evaluate.
        k: The number of neighbors to use. Defaults to 5.

    Returns:
        Evaluation results for each feature selection method.
    """
    results_dict = {}
    # iterate over all repeated experiments
    for experiment_id in repeated_experiments:
        print(f"Evaluating experiment {experiment_id}")
        file_path = os.path.join(pickle_base_path, f"{experiment_id}_result_dict.pkl")

        try:
            with open(file_path, "rb") as file:
                raw_result_dict = pickle.load(file)

                # iterate over all feature selection methods
                for key in raw_result_dict.keys():
                    if "reverse" in key or "standard" in key:
                        # select feature subsets
                        feature_importance_matrix = extract_feature_importance_matrix(raw_result_dict[key], key)
                        append_to_results_dict(
                            results_dict, key, "stability", stability_estimator.get_stability(feature_importance_matrix)
                        )
                        append_to_results_dict(
                            results_dict,
                            key,
                            "robustness_array",
                            np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=0),
                        )
                        append_to_results_dict(
                            results_dict,
                            key,
                            "subset_size_array",
                            np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=1),
                        )
                        append_to_results_dict(
                            results_dict,
                            key,
                            "mean_subset_size",
                            np.mean(np.sum(np.where(feature_importance_matrix > 0, 1, 0), axis=1), axis=0),
                        )
                        append_to_results_dict(results_dict, key, "importance_ranking", feature_importance_matrix)

                        # evaluate loo cv performance
                        y_predict, y_predict_proba, y_true = validate_cv(
                            raw_result_dict["indices"], data_df, feature_importance_matrix, k
                        )
                        append_performance_metrics(
                            key, "cv_performance", results_dict, y_predict, y_predict_proba, y_true
                        )

                        # evaluate test performance
                        # aggregate feature importance matrix
                        # TODO train complete train data without loo cv?
                        y_predict1, y_predict_proba1, y_true1 = train_model_and_predict_y(
                            data_df, test_data, feature_importance=np.sum(feature_importance_matrix, axis=0), k=k
                        )
                        append_performance_metrics(
                            key, "test_performance", results_dict, y_predict1, y_predict_proba1, y_true1
                        )
        except FileNotFoundError:
            print(f"File {file_path} not found.")
    return results_dict


def append_performance_metrics(
    feature_selection_method, performance_evaluation_method, results_dict, y_predict, y_predict_proba, y_true
):
    for metric, value in calculate_performance_metrics(y_true, y_predict, y_predict_proba).items():
        # check if cv_performance is already a key in the dictionary
        if performance_evaluation_method not in results_dict[feature_selection_method]:
            results_dict[feature_selection_method][performance_evaluation_method] = {}
        # check if the metric is already a key in the dictionary
        if metric not in results_dict[feature_selection_method][performance_evaluation_method]:
            results_dict[feature_selection_method][performance_evaluation_method][metric] = [value]
        else:
            results_dict[feature_selection_method][performance_evaluation_method][metric].append(value)


def plot_average_results(evaluated_results_dict: dict, save: bool = False, data_name: str = "Data"):
    """
    Plot the average results and the standard deviation for the given result dictionary.

    Args:
        evaluated_results_dict: The dictionary containing the evaluated results to plot.
        save: A boolean indicating whether to save the plot as a pdf. Defaults to False.
        data_name: The name of the data to plot. Defaults to "Data".
    """
    barchart_counter = 0
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
                df.loc[len(df)] = [
                    feature_selection_method,
                    np.mean(
                        evaluated_results_dict[feature_selection_method][performance_evaluation_method][
                            performance_metric
                        ]
                    ),
                    np.mean(evaluated_results_dict[feature_selection_method]["stability"]),
                    np.mean(evaluated_results_dict[feature_selection_method]["mean_subset_size"]),
                    np.std(
                        evaluated_results_dict[feature_selection_method][performance_evaluation_method][
                            performance_metric
                        ]
                    ),
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
            fig.update_yaxes(range=[0, 1.0])
            fig.update_layout(
                title=f"{data_name}: {performance_metric} vs Stability (colored and shaped by number of selected "
                f"features)",
                xaxis_title="Stability of Feature Selection",
                yaxis_title=f"{performance_metric} - {performance_evaluation_method}",
                legend_title="Average Number of Selected Features",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            # save figure as pdf
            if save:
                pio.write_image(fig, f"{performance_metric}_performance.pdf")
            fig.show()

            # plot barchart for number of selected features only once
            if not barchart_counter > 0:
                # plot results of num_features as plotly bar chart and add error bars
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=df["method"],
                        y=df["num_features"],
                        name="Number of Selected Features",
                        error_y=dict(type="data", array=df["e_num_features"]),
                    )
                )
                fig.update_layout(
                    title=f"{data_name}: Number of Selected Features",
                    xaxis_title="Feature Selection Method",
                    yaxis_title="Number of Selected Features",
                )
                fig.show()
                barchart_counter += 1


list_of_experiments = [
    "colon00",
    "colon_s10",
    "colon_s20",
    "colon_s30",
    "colon_s40",
]
test_data_df, input_data_df = load_train_test_data_for_standardized_sample_size("colon")

# list_of_experiments = [
#     "prostate_s00",
# ]
# test_data_df, input_data_df = load_data_with_standardized_sample_size("prostate")
result_dict_cv = evaluate(
    input_data_df, test_data_df, pickle_base_path=f"results", repeated_experiments=list_of_experiments, k=7
)
plot_average_results(result_dict_cv, data_name="Colon Cancer")
# TODO print table with results (tabulate, latex_raw
