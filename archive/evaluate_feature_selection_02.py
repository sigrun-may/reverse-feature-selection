# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""This module evaluates the feature selection results of different feature selection methods."""
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
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

from feature_selection_benchmark.data_loader_tools import load_train_test_data_for_standardized_sample_size
from feature_selection_benchmark.feature_selection_evaluation import stability_estimator

logging.basicConfig(level=logging.INFO)
pio.kaleido.scope.mathjax = None  # bugfix for plotly


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_importance: np.ndarray,
    random_state: int,
) -> tuple:
    """Trains a model and predicts the labels for the test data.

    Args:
        train_data: The training data.
        test_data: The test data.
        feature_importance: The feature importance.
        random_state: The random state for reproducibility.

    Returns:
        Prediction results: predicted labels, predicted probabilities, true labels.
    """
    # extract x and y values
    x_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    assert feature_importance.size == train_data.shape[1] - 1  # -1 for label column
    assert np.count_nonzero(feature_importance) > 0, "No features selected."

    # select relevant feature subset
    # extract indices of selected features where feature_importance is nonzero
    relevant_feature_subset_indices = np.flatnonzero(feature_importance)
    feature_subset_x_train = x_train[:, relevant_feature_subset_indices]
    feature_subset_x_test = x_test[:, relevant_feature_subset_indices]
    assert feature_subset_x_train.shape == (train_data.shape[0], np.count_nonzero(feature_importance))
    assert feature_subset_x_test.shape == (test_data.shape[0], np.count_nonzero(feature_importance))

    # train and evaluate the model
    clf = RandomForestClassifier(
        n_estimators=20,  # Fewer trees to avoid overfitting
        max_features=None,  # Consider all selected features at each split
        max_depth=4,  # Limit the depth of the trees
        min_samples_leaf=2,  # More samples needed at each leaf node
        bootstrap=False,  # Use the whole dataset for each tree
        class_weight="balanced",  # Adjust weights for class imbalance if any
        random_state=random_state,  # Set random seed for reproducibility
    )
    clf.fit(x_train, y_train)
    predicted_proba_y = clf.predict_proba(x_test)
    predicted_y = clf.predict(x_test)
    return predicted_y, predicted_proba_y, y_test


def calculate_performance_metrics(
    feature_importance_matrix: np.ndarray,
    random_state: int,
    experiment_data_df: pd.DataFrame,
    hold_out_data_df: pd.DataFrame = None,
) -> Dict[str, float]:
    """Calculate the performance metrics for the predicted labels.

    Args:
        feature_importance_matrix: The feature importance matrix where each row corresponds to the feature importance of a
            cross-validation iteration.
        random_state: The random state for reproducibility.
        experiment_data_df: The experiment data containing the features and labels.
        hold_out_data_df: The hold-out data for validation. If None, the test sample of the
            leave-one-out cross-validation is used.

    Returns:
        A dictionary containing the performance metrics.
    """
    if not isinstance(feature_importance_matrix, np.ndarray) or feature_importance_matrix.ndim != 2:
        raise ValueError("feature_importance_matrix must be a 2D numpy array.")
    if not isinstance(experiment_data_df, pd.DataFrame):
        raise ValueError("experiment_data_df must be a pandas DataFrame.")
    if hold_out_data_df is not None and not isinstance(hold_out_data_df, pd.DataFrame):
        raise ValueError("hold_out_data_df must be a pandas DataFrame or None.")
    assert feature_importance_matrix.shape[0] == experiment_data_df.shape[0], feature_importance_matrix.shape[0]

    performance_metrics_dict = {}
    y_predict, y_predict_proba, y_true = [], [], []

    # iterate over cross-validation indices of leave-one-out cross-validation
    for cv_iteration in range(feature_importance_matrix.shape[0]):
        feature_importances = feature_importance_matrix[cv_iteration, :]
        if np.count_nonzero(feature_importances) == 0:
            print("no features selected in iteration", cv_iteration)
            continue

        train_data = experiment_data_df.iloc[experiment_data_df.index != cv_iteration, :]
        test_data = (
            experiment_data_df.iloc[cv_iteration, :].to_frame().T if hold_out_data_df is None else hold_out_data_df
        )

        y_pred, y_pred_proba, true_y = train_and_predict(train_data, test_data, feature_importances, random_state)

        if hold_out_data_df is None:
            y_predict.extend(y_pred)
            y_predict_proba.extend(y_pred_proba)
            y_true.extend(true_y)
        else:
            hold_out_performance_metrics_dict = calculate_performance(y_pred, y_pred_proba, true_y)
            # append the performance metrics for the hold-out data to the performance metrics dictionary
            for metric, value in hold_out_performance_metrics_dict.items():
                if metric not in performance_metrics_dict.keys():
                    performance_metrics_dict[metric] = []
                performance_metrics_dict[metric].append(value)

    if hold_out_data_df is None:
        # calculate the performance metrics for the leave-one-out cross-validation
        performance_metrics_dict = calculate_performance(y_predict, y_predict_proba, y_true)
    else:
        assert len(performance_metrics_dict) > 0, "No performance metrics calculated."
        # calculate the mean performance metrics for the hold-out data
        for metric, values in performance_metrics_dict.items():
            performance_metrics_dict[metric] = np.mean(values)
    return performance_metrics_dict


def calculate_performance(y_predict, y_predict_proba, y_true) -> Dict[str, float]:
    """Calculate the performance metrics for the predicted labels.

    Args:
        y_predict: The predicted labels.
        y_predict_proba: The predicted probabilities.
        y_true: The true labels.

    Returns:
        A dictionary containing the performance metrics.
    """
    # extract the probability for the positive class for briar score calculation
    probability_positive_class = np.array([proba[1] for proba in y_predict_proba])
    performance_metrics_dict = {}
    fpr, tpr, _ = roc_curve(y_true, y_predict)
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    performance_metrics_dict["Accuracy"] = accuracy_score(y_true, y_predict)
    performance_metrics_dict["Precision"] = precision_score(y_true, y_predict)
    performance_metrics_dict["Recall"] = recall_score(y_true, y_predict)
    performance_metrics_dict["F1"] = f1_score(y_true, y_predict, average="weighted")
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
        "stability": stability_estimator.calculate_stability(input_feature_importance_matrix),
        "robustness_array": robustness_array,
        "number_of_robust_features": number_of_robust_features,
        "subset_size_array": subset_size_array,
        "mean_subset_size": mean_subset_size,
        "std_subset_size": std_subset_size,
        "cumulated_importance": np.sum(input_feature_importance_matrix, axis=0),
        "importance_matrix": input_feature_importance_matrix,
    }
    return evaluation_results_dict


def evaluate_experiment(
    input_result_dict: dict,
    experiment_data_df: pd.DataFrame,
    test_data_df: pd.DataFrame = None,
):
    """Evaluate the feature selection results of the given experiment.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        experiment_data_df: The experiment data containing the features and labels.
        test_data_df: The test data to evaluate if a hold-out data set is available.
    """
    # initialize evaluation dictionary
    input_result_dict["evaluation"] = {}
    if test_data_df is None:
        performance_evaluation_method = "performance_loo"
    else:
        performance_evaluation_method = "performance_hold_out"

    assert "standard_random_forest" in input_result_dict.keys()
    assert "standard_random_forest_meta_data" in input_result_dict.keys()
    assert "reverse_random_forest" in input_result_dict.keys()
    assert "reverse_random_forest_meta_data" in input_result_dict.keys()

    # evaluate the feature selection results for the standard random forest
    loo_cv_iteration_list = input_result_dict["standard_random_forest"]
    assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

    importance_calculation_method_list = [
        "gini_impurity",
        "gini_impurity_default_parameters",
        "permutation_importance",
        "summed_shap_values",
    ]
    # initialize feature importance matrix with zeros
    feature_importance_matrix = np.zeros((len(loo_cv_iteration_list), len(loo_cv_iteration_list[0]["gini_impurity"])))
    # evaluate each importance calculation method
    for importance_calculation_method in importance_calculation_method_list:
        # extract importance matrix
        for i, cv_iteration_result in enumerate(loo_cv_iteration_list):
            feature_importance_matrix[i, :] = cv_iteration_result[importance_calculation_method]

        # evaluate the feature subset selection results
        input_result_dict["evaluation"][importance_calculation_method] = evaluate_importance_matrix(
            feature_importance_matrix
        )
        input_result_dict["evaluation"][importance_calculation_method][performance_evaluation_method] = (
            calculate_performance_metrics(
                feature_importance_matrix,
                random_state=input_result_dict["standard_random_forest_meta_data"]["random_state"],
                experiment_data_df=experiment_data_df,
                hold_out_data_df=test_data_df,
            )
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
                    if "f_" in cv_iteration_result.index[j]:
                        feature_subset_selection_array[j] = cv_iteration_result.loc[f"f_{j}", "fraction_mean"]
                    else:
                        feature_subset_selection_array[j] = cv_iteration_result.loc[f"{j}", "fraction_mean"]
            feature_importance_matrix[i, :] = feature_subset_selection_array

        # evaluate the feature subset selection results
        input_result_dict["evaluation"]["reverse_random_forest"] = evaluate_importance_matrix(feature_importance_matrix)
        input_result_dict["evaluation"]["reverse_random_forest"][performance_evaluation_method] = (
            calculate_performance_metrics(
                feature_importance_matrix,
                random_state=input_result_dict["reverse_random_forest_meta_data"]["random_state"],
                experiment_data_df=experiment_data_df,
                hold_out_data_df=test_data_df,
            )
        )


def initialize_results_dict(feature_selection_methods, metrics):
    """Initialize a dictionary to store the performance metrics for different feature selection methods."""
    results_dict = {}
    for method in feature_selection_methods:
        if not isinstance(method, str):
            raise ValueError(f"Invalid feature selection method: {method}")
        results_dict[method] = {}
        for metric in metrics:
            if not isinstance(metric, str):
                raise ValueError(f"Invalid metric: {metric}")
            if metric.startswith("performance"):
                # insert the dictionary for the performance metrics
                results_dict[method][metric] = {"AUC": [], "Accuracy": [], "F1": [], "Brier Score": []}
            else:
                # initialize the lists for the stability and mean subset size
                results_dict[method][metric] = []
    return results_dict


def load_experiment_results(result_dict_path: str, experiment_id: str) -> Dict:
    """Load the raw experiment results from a pickle file.

    Args:
        result_dict_path: The path to dictionary of the pickled results files.
        experiment_id: The id of the experiment.

    Returns:
        The loaded experiment results.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = os.path.join(result_dict_path, f"{experiment_id}_result_dict.pkl")
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} not found.")

    with open(file_path, "rb") as file:
        result_dict = pickle.load(file)

    return result_dict


def evaluate_repeated_experiments(
    data_df: pd.DataFrame,
    test_data_df: pd.DataFrame | None,
    result_dict_path: str,
    repeated_experiments: list[str],
):
    """Evaluate the feature selection results and their reproducibility for each provided feature selection method.

    Args:
        data_df: The underlying original data for the given experiments.
        test_data_df: The test data to evaluate.
        result_dict_path: The base path to the pickled results files.
        repeated_experiments: A list of the experiment ids of repeated experiments to evaluate.

    Returns:
        Evaluation results for each feature selection method.
    """
    # define the feature selection methods and metrics
    feature_selection_methods = [
        "gini_impurity_default_parameters",
        "gini_impurity",
        "summed_shap_values",
        "permutation_importance",
        "reverse_random_forest",
    ]
    metrics = ["performance_hold_out", "stability", "mean_subset_size"]
    if test_data_df is None:
        metrics.remove("performance_hold_out")

    # initialize the results dictionary for the summarised results
    summarised_results_dict = initialize_results_dict(feature_selection_methods, metrics)

    # iterate over all repeated experiments
    for experiment_id in repeated_experiments:
        logging.info(f"Evaluating experiment {experiment_id}")
        result_dict = load_experiment_results(result_dict_path, experiment_id)

        # check if the result dictionary already contains the evaluation results
        if "evaluation" in result_dict.keys():
            logging.info(f"Evaluation results for {experiment_id} already present.")
            # delete the evaluation results to reevaluate the experiment
            del result_dict["evaluation"]

        # evaluate the experiment
        evaluate_experiment(result_dict, data_df, test_data_df)

        # extract the evaluation results and summarise them
        for feature_selection_method in result_dict["evaluation"].keys():
            # iterate over the evaluation metrics
            for metric in result_dict["evaluation"][feature_selection_method].keys():
                if metric.startswith("performance"):
                    # iterate over the performance metrics
                    for performance_metric in summarised_results_dict[feature_selection_method][metric].keys():
                        summarised_results_dict[feature_selection_method][metric][performance_metric].append(
                            result_dict["evaluation"][feature_selection_method][metric][performance_metric]
                        )
                elif metric == "stability" or metric == "mean_subset_size":
                    # append stability and mean subset size
                    summarised_results_dict[feature_selection_method][metric].append(
                        result_dict["evaluation"][feature_selection_method][metric]
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


def plot_average_results(
    evaluated_results_dict: dict,
    save: bool = False,
    path: str = "",
    data_display_name: str = "Data",
    hold_out: bool = False,
    loo: bool = True,
):
    """Plot the average results and the standard deviation for the given result dictionary.

    Args:
        evaluated_results_dict: The dictionary containing the evaluated results to plot.
        save: A boolean indicating whether to save the plot as a pdf. Defaults to False.
        path: The path to save the plot. Defaults to an empty string.
        data_display_name: The name of the data to plot. Defaults to "Data".
        hold_out: A boolean indicating whether to plot the hold-out data. Defaults to False.
        loo: A boolean indicating whether to plot the leave-one-out cross-validation data. Defaults to True.
    """
    barchart_counter = 0
    performance_evaluation_methods = []
    if loo:
        performance_evaluation_methods.append("performance_loo")
    if hold_out:
        performance_evaluation_methods.append("performance_hold_out")
    assert len(performance_evaluation_methods) > 0, "No performance evaluation method selected."

    # initialize data frame for the latex table
    df_for_latex = pd.DataFrame(
        columns=[
            "method",
            "num features",
            "AUC",
            "Accuracy",
            "F1",
            "Brier Score",
            "stability",
            "e_num_features",
            "e_AUC",
            "e_Accuracy",
            "e_F1",
            "e_Brier Score",
            "e_stability",
        ]
    )
    # prepare df for latex table

    for feature_selection_method in evaluated_results_dict.keys():
        # print("feature_selection_method", feature_selection_method)
        assert len(evaluated_results_dict[feature_selection_method]["mean_subset_size"]) == 3
        assert len(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["AUC"]) == 3
        assert len(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["Accuracy"]) == 3
        assert len(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["F1"]) == 3
        assert (
            len(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["Brier Score"]) == 3
        )
        assert len(evaluated_results_dict[feature_selection_method]["stability"]) == 3

        df_for_latex.loc[len(df_for_latex)] = [
            feature_selection_method,
            np.mean(evaluated_results_dict[feature_selection_method]["mean_subset_size"]),
            np.mean(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["AUC"]),
            np.mean(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["Accuracy"]),
            np.mean(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["F1"]),
            np.mean(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["Brier Score"]),
            np.mean(evaluated_results_dict[feature_selection_method]["stability"]),
            np.std(evaluated_results_dict[feature_selection_method]["mean_subset_size"]),
            np.std(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["AUC"]),
            np.std(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["Accuracy"]),
            np.std(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["F1"]),
            np.std(evaluated_results_dict[feature_selection_method][performance_evaluation_methods[0]]["Brier Score"]),
            np.std(evaluated_results_dict[feature_selection_method]["stability"]),
        ]

    # round floats to five decimal places
    df_for_latex["num features"] = df_for_latex["num features"].apply(lambda x: round(x, 5))
    df_for_latex["AUC"] = df_for_latex["AUC"].apply(lambda x: round(x, 5))
    df_for_latex["Accuracy"] = df_for_latex["Accuracy"].apply(lambda x: round(x, 5))
    df_for_latex["F1"] = df_for_latex["F1"].apply(lambda x: round(x, 5))
    df_for_latex["Brier Score"] = df_for_latex["Brier Score"].apply(lambda x: round(x, 5))
    df_for_latex["stability"] = df_for_latex["stability"].apply(lambda x: round(x, 5))
    df_for_latex["e_num_features"] = df_for_latex["e_num_features"].apply(lambda x: round(x, 5))
    df_for_latex["e_AUC"] = df_for_latex["e_AUC"].apply(lambda x: round(x, 5))
    df_for_latex["e_Accuracy"] = df_for_latex["e_Accuracy"].apply(lambda x: round(x, 5))
    df_for_latex["e_F1"] = df_for_latex["e_F1"].apply(lambda x: round(x, 5))
    df_for_latex["e_Brier Score"] = df_for_latex["e_Brier Score"].apply(lambda x: round(x, 5))
    df_for_latex["e_stability"] = df_for_latex["e_stability"].apply(lambda x: round(x, 5))

    # append latex command /pm to the error values
    df_for_latex["e_stability"] = df_for_latex["e_stability"].apply(lambda x: f"$\pm{x}$")
    df_for_latex["e_AUC"] = df_for_latex["e_AUC"].apply(lambda x: f"$\pm{x}$")
    df_for_latex["e_Accuracy"] = df_for_latex["e_Accuracy"].apply(lambda x: f"$\pm{x}$")
    df_for_latex["e_F1"] = df_for_latex["e_F1"].apply(lambda x: f"$\pm{x}$")
    df_for_latex["e_Brier Score"] = df_for_latex["e_Brier Score"].apply(lambda x: f"$\pm{x}$")
    df_for_latex["e_num_features"] = df_for_latex["e_num_features"].apply(lambda x: f"$\pm{x}$")

    # # round floats
    # df_for_latex["num features"] = df_for_latex["num features"].apply(lambda x: round(x, 2))
    # df_for_latex["AUC"] = df_for_latex["AUC"].apply(lambda x: round(x, 2))
    # df_for_latex["Accuracy"] = df_for_latex["Accuracy"].apply(lambda x: round(x, 2))
    # df_for_latex["F1"] = df_for_latex["F1"].apply(lambda x: round(x, 2))
    # df_for_latex["Brier Score"] = df_for_latex["Brier Score"].apply(lambda x: round(x, 2))
    # df_for_latex["stability"] = df_for_latex["stability"].apply(lambda x: round(x, 2))

    # merge the columns for performance and stability and the respective standard error
    df_for_latex["stability"] = df_for_latex["stability"].astype(str) + " " + df_for_latex["e_stability"]
    df_for_latex["AUC"] = df_for_latex["AUC"].astype(str) + " " + df_for_latex["e_AUC"]
    df_for_latex["Accuracy"] = df_for_latex["Accuracy"].astype(str) + " " + df_for_latex["e_Accuracy"]
    df_for_latex["F1"] = df_for_latex["F1"].astype(str) + " " + df_for_latex["e_F1"]
    df_for_latex["Brier Score"] = df_for_latex["Brier Score"].astype(str) + " " + df_for_latex["e_Brier Score"]
    df_for_latex["num features"] = df_for_latex["num features"].astype(str) + " " + df_for_latex["e_num_features"]

    df_for_latex["method"] = [
        "random forest without HPO",
        "gini importance",
        "shap values",
        "permutation importance",
        "reverse random forest",
    ]

    print(
        df_for_latex.to_latex(
            index=False,
            columns=["method", "num features"],
            header=[
                "",
                f"Number of Selected Features",
            ],
            label=f"table:{data_display_name}",
            caption=(f"{data_display_name}: Performance Metrics and Stability of Feature Selection", "table"),
        )
    )

    print(
        df_for_latex.to_latex(
            index=False,
            columns=["method", "stability"],
            header=[
                "",
                "Stability of Feature Selection",
            ],
        )
    )

    for performance_metric in ["AUC", "Accuracy", "F1", "Brier Score"]:
        print(
            df_for_latex.to_latex(
                index=False,
                columns=["method", performance_metric],
                header=[
                    "",
                    performance_metric,
                ],
            )
        )

    # prepare scatter plots for each performance evaluation method
    for performance_evaluation_method in performance_evaluation_methods:
        for performance_metric in ["AUC", "Accuracy", "F1", "Brier Score"]:
            # Create a DataFrame
            plot_df = pd.DataFrame(
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
            for feature_selection_method in evaluated_results_dict:
                # print("feature_selection_method", feature_selection_method)
                plot_df.loc[len(plot_df)] = [
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
                plot_df,
                x="stability",
                y="performance",
                error_y="e_performance",
                error_x="e_stability",
                # size="num_features",
                # color="num_features",
                hover_data=["method"],
                text=[
                    "without HPO",
                    "gini",
                    "shap",
                    "permutation",
                    "reverse random forest",
                ],  # "method",  # Add method name to each data point
                template="plotly_white",
                height=350,
            )
            # unoptimized, gini, shap, permutation, reverse
            if data_display_name == "Leukemia":
                fig.update_traces(textposition=["top left", "top right", "bottom right", "bottom center", "top center"])
            elif "olon" in data_display_name:
                fig.update_traces(textposition=["top right", "top center", "bottom center", "top left", "bottom right"])
            else:
                fig.update_traces(textposition=["top right", "top left", "bottom left", "top left", "bottom right"])
            fig.update_xaxes(range=[0, 1.01])
            fig.update_yaxes(range=[0, 1.01])
            fig.update_layout(
                # title=f"{data_display_name}: {performance_metric} vs Stability",
                xaxis_title="Stability of Feature Selection",
                yaxis_title=f"{data_display_name}: {performance_metric}",
                legend_title="Average Number of Selected Features",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            # save figure as pdf
            if save:
                plot_path = f"{path}/{data_display_name}_{performance_metric}_{performance_evaluation_method}.pdf"
                pio.write_image(fig, plot_path)
            fig.show()

            # plot barchart for number of selected features only once
            if not barchart_counter > 0:
                plot_df["method"] = [
                    "random forest without HPO",
                    "gini importance",
                    "shap values",
                    "permutation importance",
                    "reverse random forest",
                ]

                # append latex command /pm to the error values
                plot_df["e_num_features"] = plot_df["e_num_features"].apply(lambda x: f"$\pm{round(x,2)}$")

                print(
                    plot_df.to_latex(
                        index=False,
                        columns=["method", "num_features", "e_num_features"],
                        header=[
                            data_display_name,
                            "Number of Selected Features",
                            "Error",
                        ],
                        float_format="{:.2f}".format,
                    )
                )

                # # plot results of num_features as plotly bar chart and add error bars
                # fig = go.Figure()
                # fig.add_trace(
                #     go.Bar(
                #         x=df["method"],
                #         y=df["num_features"],
                #         name=f"{data_display_name}: Number of Selected Features",
                #         error_y=dict(type="data", array=df["e_num_features"]),
                #         text=round(df["num_features"], 2),
                #         textposition="inside",
                #     )
                # )
                # print(data_display_name)
                # fig.update_layout(
                #     # title=f"{data_display_name}: Number of Selected Features",
                #     xaxis_title="Feature Selection Method",
                #     yaxis_title=f"{data_display_name}: Number of Selected Features",
                #     margin=dict(l=20, r=20, t=20, b=20),
                # )
                # # save figure as pdf
                # if save:
                #     plot_path = f"{path}/{data_display_name}_num_features.pdf"
                #     pio.write_image(fig, plot_path)
                barchart_counter += 1


base_path = "../results"
# list_of_experiments = [
#     "colon_s10",
#     "colon_s20",
#     # "colon_s30",
#     # "colon_s40",
#     "colon_shift_00",
# ]
# data_name = "colon"
# hold_out_df, input_df = load_train_test_data_for_standardized_sample_size(data_name)
# if data_name != "leukemia_big":
#     data_name = f"{data_name.capitalize()} Cancer"
# else:
#     data_name = "Leukemia"
#
# plot_average_results(
#     evaluate_repeated_experiments(input_df, hold_out_df, base_path, list_of_experiments),
#     save=True,
#     path=f"{base_path}/images",
#     data_display_name=data_name,
#     hold_out=True,
#     loo=False,
# )
#
# list_of_experiments = [
#     "prostate_shift_00",
#     "prostate_shift_01",
#     "prostate_shift_02",
# ]
# data_name = "prostate"
# hold_out_df, input_df = load_train_test_data_for_standardized_sample_size(data_name)
# if data_name != "leukemia_big":
#     data_name = f"{data_name.capitalize()} Cancer"
# else:
#     data_name = "Leukemia"
#
# plot_average_results(
#     evaluate_repeated_experiments(input_df, hold_out_df, base_path, list_of_experiments),
#     save=True,
#     path=f"{base_path}/images",
#     data_display_name=data_name,
#     hold_out=True,
#     loo=False,
# )
list_of_experiments = [
    "leukemia_shift_00",
    "leukemia_big_shift10",
    "leukemia_shift_02",
]
data_name = "leukemia_big"

hold_out_df, input_df = load_train_test_data_for_standardized_sample_size(data_name)
data_name = f"{data_name.capitalize()} Cancer" if data_name != "leukemia_big" else "Leukemia"

plot_average_results(
    evaluate_repeated_experiments(input_df, hold_out_df, base_path, list_of_experiments),
    save=True,
    path=f"{base_path}/images",
    data_display_name=data_name,
    hold_out=True,
    loo=False,
)

# result_dict_path = Path("../../results/random_noise_lognormal_30_7000_01_result_dict.pkl")
# list_of_experiments = ["random_noise_lognormal_30_7000_01"]
# input_df = pd.read_csv("../../data/random_noise_lognormal_30_7000_01_data_df.csv")
# hold_out_df = None
# assert input_df.columns[0] == "label", input_df.columns[0]

# data_df.drop(columns=[data_df.columns[0]], inplace=True)
# input_data_df.to_csv("../../results/random_noise_lognormal_30_7000_01_data_df.csv", index=False)
