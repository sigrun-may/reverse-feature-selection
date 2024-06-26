# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""This module evaluates the feature selection results of different feature selection methods."""
import os
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
    log_loss,
    brier_score_loss,
    average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer

from feature_selection_benchmark.data_loader_tools import load_train_test_data_for_standardized_sample_size
from feature_selection_benchmark.feature_selection_evaluation import stability_estimator
from feature_selection_benchmark.feature_selection_evaluation.weighted_manhattan_distance import (
    WeightedManhattanDistance,
)


def train_and_predict(
        train_data: pd.DataFrame, test_data: pd.DataFrame, feature_selection_result: np.ndarray, k: int
) -> tuple:
    """Trains a model and predicts the labels for the test data.

    Args:
        train_data: The training data.
        test_data: The test data.
        feature_selection_result: The feature importance.
        k: If None, a random forest classifier with standard hyperparameters is used.
            Else, the number of neighbors to use for KNN.

    Returns:
        Prediction results: predicted labels, predicted probabilities, true labels.
    """
    # extract x and y values
    x_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # feature_subset_weights = feature_importance
    assert (
            feature_selection_result.size == train_data.shape[1] - 1
    ), feature_selection_result.size  # -1 for label column
    assert np.count_nonzero(feature_selection_result) > 0, np.count_nonzero(feature_selection_result)

    # select relevant feature subset
    # extract indices of selected features where feature_subset_weights is nonzero
    relevant_feature_subset_indices = np.flatnonzero(feature_selection_result)
    feature_subset_x_train = x_train[:, relevant_feature_subset_indices]
    assert feature_subset_x_train.shape == (
        train_data.shape[0],
        np.count_nonzero(feature_selection_result),
    ), feature_subset_x_train.shape
    feature_subset_x_test = x_test[:, relevant_feature_subset_indices]
    assert feature_subset_x_test.shape == (
        test_data.shape[0],
        np.count_nonzero(feature_selection_result),
    ), f"{feature_subset_x_test.shape}, {test_data.shape[0]}, {np.count_nonzero(feature_selection_result)}"

    # train and evaluate the model
    if k is None:
        # print("train random forest classifier")
        clf = RandomForestClassifier()
    else:
        # scale data: choose quantile scaler due to the presence of outliers
        robust_scaler = QuantileTransformer(n_quantiles=feature_subset_x_train.shape[0], output_distribution="normal")

        x_train = robust_scaler.fit_transform(feature_subset_x_train)
        x_test = robust_scaler.transform(feature_subset_x_test)
        feature_subset_weights = feature_selection_result[relevant_feature_subset_indices]
        assert feature_subset_weights.size == np.count_nonzero(feature_selection_result), feature_subset_weights.size
        # print("use weighted KNeighborsClassifier")
        clf = KNeighborsClassifier(
            n_neighbors=k,
            metric=WeightedManhattanDistance(feature_subset_weights),
        )
        # # print("use KNeighborsClassifier")
        # clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    clf.fit(x_train, y_train)
    predicted_proba_y = clf.predict_proba(x_test)
    predicted_y = clf.predict(x_test)
    return predicted_y, predicted_proba_y, y_test


def calculate_performance_metrics(feature_importance_matrix: np.ndarray,
                                  experiment_data_df: pd.DataFrame,
                                  hold_out_data_df: pd.DataFrame = None,
                                  k: int = None) -> Dict[str, float]:
    """Calculate the performance metrics for the predicted labels.

    Args:
        feature_importance_matrix: The feature importance matrix where each row corresponds to the feature importance of a
            cross-validation iteration.
        experiment_data_df: The experiment data containing the features and labels.
        hold_out_data_df: The hold-out data for validation. If None, the test sample of the
            leave-one-out cross-validation is used.
        k: The number of neighbors to use for KNN. If None, a random forest classifier with standard hyperparameters
            is used.

    Returns:
        A dictionary containing the performance metrics.
    """
    # predict y values
    y_predict = []
    y_predict_proba = []
    y_true = []
    # iterate over cross-validation indices of leave-one-out cross-validation
    for cv_iteration in range(experiment_data_df.shape[0]):
        feature_importances = feature_importance_matrix[cv_iteration, :]
        if np.count_nonzero(feature_importances) == 0:
            print("no features selected in iteration", cv_iteration)
            continue
        # extract training and test data samples
        train_data = experiment_data_df.iloc[experiment_data_df.index != cv_iteration, :]
        if hold_out_data_df is None:
            # use test sample of leave one out cross-validation for testing
            test_data = experiment_data_df.iloc[cv_iteration, :].to_frame().T
        else:
            # use hold-out data for validation
            test_data = hold_out_data_df
        y_pred, y_pred_proba, true_y = train_and_predict(train_data, test_data, feature_importances, k)
        y_predict.extend(y_pred)
        y_predict_proba.extend(y_pred_proba)
        y_true.extend(true_y)

    # extract the probability for the positive class for briar score calculation
    probability_positive_class = np.array([proba[1] for proba in y_predict_proba])

    performance_metrics_dict = {}
    fpr, tpr, _ = roc_curve(y_true, y_predict)
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    performance_metrics_dict["Accuracy"] = accuracy_score(y_true, y_predict)
    performance_metrics_dict["Precision"] = precision_score(y_true, y_predict)
    performance_metrics_dict["Recall"] = recall_score(y_true, y_predict)
    performance_metrics_dict["F1"] = f1_score(y_true, y_predict)
    performance_metrics_dict["AUC"] = auc(fpr, tpr)
    performance_metrics_dict["Log Loss"] = log_loss(y_true, y_predict_proba)
    performance_metrics_dict["Brier Score"] = brier_score_loss(y_true, probability_positive_class)
    performance_metrics_dict["AUPRC"] = average_precision_score(y_true, y_predict)
    return performance_metrics_dict


def evaluate_importance_matrix(input_feature_importance_matrix: np.ndarray) -> Dict[str, Any]:
    """Evaluate the feature importance matrix.

    Args:
        input_feature_importance_matrix: The feature importance matrix where each row corresponds to the
            feature importance of a cross-validation iteration.

    Returns:
        A dictionary containing the evaluation results.
    """
    robustness_array = np.sum(np.where(input_feature_importance_matrix > 0, 1, 0), axis=0)
    number_of_robust_features = np.sum(np.where(robustness_array == input_feature_importance_matrix.shape[0], 1, 0))
    subset_size_array = np.sum(np.where(input_feature_importance_matrix > 0, 1, 0), axis=1)
    mean_subset_size = np.mean(subset_size_array)
    std_subset_size = np.std(subset_size_array)

    evaluation_results_dict = {
        "stability": stability_estimator.get_stability(input_feature_importance_matrix),
        "robustness_array": robustness_array,
        "number_of_robust_features": number_of_robust_features,
        "subset_size_array": subset_size_array,
        "mean_subset_size": mean_subset_size,
        "std_subset_size": std_subset_size,
        "cumulated_importance": np.sum(input_feature_importance_matrix, axis=0),
        "importance_matrix": input_feature_importance_matrix,
    }
    return evaluation_results_dict


def evaluate_feature_selection_results(
        experiment_data_df, feature_importance_matrix, importance_calculation_method, evaluation_result_dict, k,
        test_data_df
):
    """Evaluate the feature selection results for the given feature importance matrix.

    Args:
        experiment_data_df: The experiment data containing the features and labels.
        feature_importance_matrix: The feature importance matrix where each row corresponds to the feature importance of a
            cross-validation iteration.
        importance_calculation_method: The method used for the feature selection.
        evaluation_result_dict: The dictionary to save the evaluation results.
        k: The number of neighbors to use for KNN. If None, a random forest classifier with standard hyperparameters
            is used.
        test_data_df: The test data to evaluate if a hold-out data set is available.
    """
    # evaluate the feature subset selection results
    evaluation_result_dict[importance_calculation_method] = evaluate_importance_matrix(feature_importance_matrix)
    # evaluate performance of the leave-one-out cross-validation
    evaluation_result_dict[importance_calculation_method]["performance_loo"] = calculate_performance_metrics(
        feature_importance_matrix, experiment_data_df, k=k
    )
    # evaluate performance of the hold-out data
    evaluation_result_dict[importance_calculation_method]["performance_hold_out"] = calculate_performance_metrics(
        feature_importance_matrix, experiment_data_df, hold_out_data_df=test_data_df, k=k)


def evaluate_experiment(
        input_result_dict: dict, experiment_data_df: pd.DataFrame, test_data_df: pd.DataFrame = None, k: int = None
):
    """Evaluate the feature selection results of the given experiment.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        experiment_data_df: The experiment data containing the features and labels.
        test_data_df: The test data to evaluate if a hold-out data set is available.
        k: The number of neighbors to use for KNN. If None, a random forest classifier with standard hyperparameters
            is used.
    """
    if k is None:
        print("Evaluate feature selection results without KNN classifier - RF")
    else:
        print(f"Evaluate feature selection results with KNN classifier with k={k}")
    # initialize evaluation dictionary
    input_result_dict["evaluation"] = {}

    # evaluate the feature selection results for the standard random forest
    if "standard_random_forest" in input_result_dict.keys():
        loo_cv_iteration_list = input_result_dict["standard_random_forest"]
        assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

        importance_calculation_method_list = [
            "gini_importance",
            "gini_rf_unoptimized_importance",
            "permutation_importance",
            "summed_shap_values",
        ]
        # initialize feature importance matrix with zeros
        feature_importance_matrix = np.zeros(
            (len(loo_cv_iteration_list), len(loo_cv_iteration_list[0]["gini_importance"]))
        )
        # evaluate each importance calculation method
        for importance_calculation_method in importance_calculation_method_list:
            # extract importance matrix
            for i, cv_iteration_result in enumerate(loo_cv_iteration_list):
                feature_importance_matrix[i, :] = cv_iteration_result[importance_calculation_method]

            evaluate_feature_selection_results(
                experiment_data_df,
                feature_importance_matrix,
                f"standard_random_forest_{importance_calculation_method}",
                input_result_dict["evaluation"],
                k,
                test_data_df,
            )

    # evaluate the feature selection results for the reverse random forest
    if "reverse_random_forest" in input_result_dict.keys():
        loo_cv_iteration_list = input_result_dict["reverse_random_forest"]

        # initialize feature importance matrix with zeros
        feature_importance_matrix = np.zeros((len(loo_cv_iteration_list), len(loo_cv_iteration_list[0])))
        # extract importance matrix
        for i, cv_iteration_result in enumerate(loo_cv_iteration_list):
            feature_subset_selection_array = np.zeros(len(cv_iteration_result))

            # extract feature selection array
            for j, p_value in enumerate(cv_iteration_result["p_value"]):
                # check if the value of the "p_value" column is smaller or equal than 0.05
                if p_value is not None and p_value <= 0.05:
                    # extract the fraction of selected features
                    feature_subset_selection_array[j] = cv_iteration_result.loc[f"f_{j}", "fraction_mean"]
            feature_importance_matrix[i, :] = feature_subset_selection_array

        evaluate_feature_selection_results(experiment_data_df, feature_importance_matrix, "reverse_random_forest",
                                           input_result_dict["evaluation"], k, test_data_df)


def evaluate_repeated_experiments(
        data_df: pd.DataFrame,
        test_data_df: pd.DataFrame,
        result_dict_path: Path,
        repeated_experiments: list[str],
        k: int | None,
):
    """Evaluate the feature selection results and their reproducibility for each provided feature selection method.

    Args:
        data_df: The underlying original data for the given experiments.
        test_data_df: The test data to evaluate.
        result_dict_path: The base path to the pickled results files.
        repeated_experiments: A list of the experiment ids of repeated experiments to evaluate.
        k: The number of neighbors to use for KNN. If None, a random forest classifier
            with standard hyperparameters is used.

    Returns:
        Evaluation results for each feature selection method.
    """
    summarised_results_dict = {}
    list_of_methods = ['standard_random_forest_gini_importance', 'standard_random_forest_gini_rf_unoptimized_importance', 'standard_random_forest_permutation_importance', 'standard_random_forest_summed_shap_values', 'reverse_random_forest']
    for fs_method in list_of_methods:
        summarised_results_dict[fs_method] = {
            "performance_loo": {
                "AUC": [],
                "Accuracy": [],
                "F1": [],
                "Brier Score": [],
            },
            "performance_hold_out": {
                "AUC": [],
                "Accuracy": [],
                "F1": [],
                "Brier Score": [],
            },
            "stability": [],
            "mean_subset_size": [],
        }
    # iterate over all repeated experiments
    for experiment_id in repeated_experiments:
        print(f"Evaluating experiment {experiment_id}")
        file_path = os.path.join(result_dict_path, f"{experiment_id}_result_dict.pkl")

        # analyze experiment results
        try:
            # load the raw experiment results
            with open(file_path, "rb") as file:
                result_dict = pickle.load(file)
        except FileNotFoundError:
            print(f"File {file_path} not found.")

        # check if the result dictionary already contains the evaluation results
        if "evaluation" in result_dict.keys():
            print(f"Evaluation results for {experiment_id} already present.")
            # continue
            # delete the evaluation results to reevaluate the experiment
            del result_dict["evaluation"]
            evaluate_experiment(result_dict, data_df, test_data_df, k=k)
        else:
            # evaluate the experiment
            evaluate_experiment(result_dict, data_df, test_data_df, k=k)

        # extract the evaluation results and store them in the summarised results dictionary
        for method in result_dict["evaluation"].keys():
            summarised_results_dict[method]["performance_loo"]["AUC"].append(
                result_dict["evaluation"][method]["performance_loo"]["AUC"]
            )
            summarised_results_dict[method]["performance_loo"]["Accuracy"].append(
                result_dict["evaluation"][method]["performance_loo"]["Accuracy"]
            )
            summarised_results_dict[method]["performance_loo"]["F1"].append(
                result_dict["evaluation"][method]["performance_loo"]["F1"]
            )
            summarised_results_dict[method]["performance_loo"]["Brier Score"].append(
                result_dict["evaluation"][method]["performance_loo"]["Brier Score"]
            )
            summarised_results_dict[method]["stability"].append(result_dict["evaluation"][method]["stability"])
            summarised_results_dict[method]["mean_subset_size"].append(result_dict["evaluation"][method]["mean_subset_size"])

            # check if the performance of an independent test set is available
            if "performance_hold_out" in result_dict["evaluation"][method].keys():
                summarised_results_dict[method]["performance_hold_out"]["AUC"].append(
                    result_dict["evaluation"][method]["performance_hold_out"]["AUC"]
                )
                summarised_results_dict[method]["performance_hold_out"]["Accuracy"].append(
                    result_dict["evaluation"][method]["performance_hold_out"]["Accuracy"]
                )
                summarised_results_dict[method]["performance_hold_out"]["F1"].append(
                    result_dict["evaluation"][method]["performance_hold_out"]["F1"]
                )
                summarised_results_dict[method]["performance_hold_out"]["Brier Score"].append(
                    result_dict["evaluation"][method]["performance_hold_out"]["Brier Score"]
                )
    # print the average results
    for method in summarised_results_dict.keys():
        print("\n")
        print(method)
        for element, value in summarised_results_dict[method].items():
            if element == "importance_matrix" or element == "robustness_array" or element == "cumulated_importance":
                continue
            print(element, ": ", value)


    # for method in result_dict["evaluation"].keys():
    #     if "shap" in method:
    #         continue
    #     print("\n")
    #     print(method)
    #     for element, value in result_dict["evaluation"][method].items():
    #         if element == "importance_matrix" or element == "robustness_array" or element == "cumulated_importance":
    #             continue
    #         print(element, ": ", value)
    return summarised_results_dict


def plot_average_results(evaluated_results_dict: dict, save: bool = False, data_name: str = "Data"):
    """Plot the average results and the standard deviation for the given result dictionary.

    Args:
        evaluated_results_dict: The dictionary containing the evaluated results to plot.
        save: A boolean indicating whether to save the plot as a pdf. Defaults to False.
        data_name: The name of the data to plot. Defaults to "Data".
    """
    barchart_counter = 0
    for performance_evaluation_method in ["performance_loo", "performance_hold_out"]:
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


base_path = Path("../../results")
# result_dict_path = Path("../../results/colon_shift_00_result_dict.pkl")
# hold_out_df, input_df = load_train_test_data_for_standardized_sample_size("colon")

list_of_experiments = [
    "prostate_shift_00", "prostate_shift_01", "prostate_shift_02",
]
hold_out_df, input_df = load_train_test_data_for_standardized_sample_size("prostate")

# list_of_experiments = [
#     "leukemia_shift_00",
# ]
# hold_out_df, input_df = load_train_test_data_for_standardized_sample_size("leukemia_big")

# result_dict_path = Path("../../results/random_noise_lognormal_30_7000_01_result_dict.pkl")
# data_df = pd.read_csv("../../results/random_noise_lognormal_30_7000_01_data_df.csv")
# hold_out_test_data_df = None
# assert input_data_df.columns[0] == "label", input_data_df.columns[0]

# data_df.drop(columns=[data_df.columns[0]], inplace=True)
# input_data_df.to_csv("../../results/random_noise_lognormal_30_7000_01_data_df.csv", index=False)
# load the raw experiment results
# with open(Path("../../results/colon_shift_00_result_dict.pkl"), "rb") as file:
#     result_dict = pickle.load(file)
# evaluate_experiment(result_dict, input_df, hold_out_df, k=None)
# print(result_dict["evaluation"].keys())
# evaluate_repeated_experiments(input_df, hold_out_df, base_path, list_of_experiments, k=None)
plot_average_results(evaluate_repeated_experiments(input_df, hold_out_df, base_path, list_of_experiments, k=None), save=False, data_name="Prostate")

# if result_dict_path.exists():
#     with open(result_dict_path, "rb") as file:
#         result_dict = pickle.load(file)
#
# evaluate_selected_feature_subsets(result_dict, k=None, test_data_df=hold_out_test_data_df)

# for method in result_dict["evaluation"].keys():
#     if "shap" in method:
#         continue
#     print("\n")
#     print(method)
#     for element, value in result_dict["evaluation"][method].items():
#         if element == "importance_matrix" or element == "robustness_array" or element == "cumulated_importance":
#             continue
#         print(element, ": ", value)
