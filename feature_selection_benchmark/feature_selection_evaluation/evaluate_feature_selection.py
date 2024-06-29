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
from typing import Dict, Any

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
from feature_selection_benchmark.data_loader_tools import load_train_test_data_for_standardized_sample_size
from feature_selection_benchmark.feature_selection_evaluation import stability_estimator

logging.basicConfig(level=logging.INFO)


def train_and_predict(
        train_data: pd.DataFrame, test_data: pd.DataFrame, feature_selection_result: np.ndarray,
) -> tuple:
    """Trains a model and predicts the labels for the test data.

    Args:
        train_data: The training data.
        test_data: The test data.
        feature_selection_result: The feature importance.

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
    clf = RandomForestClassifier(
        n_estimators=20,  # Fewer trees to avoid overfitting
        max_features=None,  # Consider all selected features at each split
        max_depth=4,  # Limit the depth of the trees
        min_samples_leaf=2,  # More samples needed at each leaf node
        bootstrap=False,  # Use the whole dataset for each tree
        class_weight="balanced",  # Adjust weights for class imbalance if any
    )
    clf.fit(x_train, y_train)
    predicted_proba_y = clf.predict_proba(x_test)
    predicted_y = clf.predict(x_test)
    return predicted_y, predicted_proba_y, y_test


def calculate_performance_metrics(
        feature_importance_matrix: np.ndarray,
        experiment_data_df: pd.DataFrame,
        hold_out_data_df: pd.DataFrame = None,
) -> Dict[str, float]:
    """Calculate the performance metrics for the predicted labels.

    Args:
        feature_importance_matrix: The feature importance matrix where each row corresponds to the feature importance of a
            cross-validation iteration.
        experiment_data_df: The experiment data containing the features and labels.
        hold_out_data_df: The hold-out data for validation. If None, the test sample of the
            leave-one-out cross-validation is used.

    Returns:
        A dictionary containing the performance metrics.
    """
    assert feature_importance_matrix.shape[0] == experiment_data_df.shape[0], feature_importance_matrix.shape[0]
    # predict y values
    y_predict = []
    y_predict_proba = []
    y_true = []
    # iterate over cross-validation indices of leave-one-out cross-validation
    for cv_iteration in range(feature_importance_matrix.shape[0]):
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
        y_pred, y_pred_proba, true_y = train_and_predict(train_data, test_data, feature_importances)
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


def evaluate_feature_selection_performance(
        evaluation_result_dict,
        importance_calculation_method,
        feature_importance_matrix,
        experiment_data_df,
        test_data_df,
):
    """Evaluate the feature selection results for the given feature importance matrix.

    Args:
        evaluation_result_dict: The dictionary to save the evaluation results.
        importance_calculation_method: The method used for the feature selection.
        feature_importance_matrix: The feature importance matrix where each row corresponds to the feature importance
            of a cross-validation iteration.
        experiment_data_df: The experiment data containing the features and labels.
        test_data_df: The test data to evaluate if a hold-out data set is available.
    """

    # evaluate performance of the leave-one-out cross-validation
    evaluation_result_dict[importance_calculation_method]["performance_loo"] = calculate_performance_metrics(
        feature_importance_matrix, experiment_data_df, hold_out_data_df=None,
    )
    if test_data_df is not None:
        # evaluate performance of the hold-out data
        evaluation_result_dict[importance_calculation_method]["performance_hold_out"] = calculate_performance_metrics(
            feature_importance_matrix, experiment_data_df, hold_out_data_df=test_data_df,
        )


def evaluate_experiment(
        input_result_dict: dict, experiment_data_df: pd.DataFrame, test_data_df: pd.DataFrame = None,
):
    """Evaluate the feature selection results of the given experiment.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        experiment_data_df: The experiment data containing the features and labels.
        test_data_df: The test data to evaluate if a hold-out data set is available.
    """
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

            # evaluate the feature subset selection results
            input_result_dict["evaluation"][importance_calculation_method] = evaluate_importance_matrix(
                feature_importance_matrix)

            evaluate_feature_selection_performance(
                evaluation_result_dict=input_result_dict["evaluation"],
                importance_calculation_method=importance_calculation_method,
                feature_importance_matrix=feature_importance_matrix,
                experiment_data_df=experiment_data_df,
                test_data_df=test_data_df,
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

        # evaluate the feature subset selection results
        input_result_dict["evaluation"]["reverse_random_forest"] = evaluate_importance_matrix(
            feature_importance_matrix)

        evaluate_feature_selection_performance(
            evaluation_result_dict=input_result_dict["evaluation"],
            importance_calculation_method="reverse_random_forest",
            feature_importance_matrix=feature_importance_matrix,
            experiment_data_df=experiment_data_df,
            test_data_df=test_data_df,
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
        "gini_rf_unoptimized_importance",
        "gini_importance",
        "summed_shap_values",
        "permutation_importance",
        "reverse_random_forest",
    ]
    metrics = ["performance_loo", "performance_hold_out", "stability", "mean_subset_size"]
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
        evaluated_results_dict: dict, save: bool = False, data_display_name: str = "Data", hold_out: bool = False,
        loo: bool = True
):
    """Plot the average results and the standard deviation for the given result dictionary.

    Args:
        evaluated_results_dict: The dictionary containing the evaluated results to plot.
        save: A boolean indicating whether to save the plot as a pdf. Defaults to False.
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
    for performance_evaluation_method in performance_evaluation_methods:
        print("plot", performance_evaluation_method)
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
                # print("feature_selection_method", feature_selection_method)
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
                text=["without HPO", "gini", "shap", "permutation", "reverse"],# "method",  # Add method name to each data point
                template="plotly_white",
            )
            # unoptimized, gini, shap, permutation, reverse
            fig.update_traces(textposition=["top right", "top left", "bottom left", "top left", "bottom right"])
            fig.update_xaxes(range=[0, 1.1])
            fig.update_yaxes(range=[0, 1.1])
            fig.update_layout(
                title=f"{data_display_name}: {performance_metric} vs Stability (colored and shaped by number of selected "
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
                print(data_display_name)
                fig.update_layout(
                    title=f"{data_display_name}: Number of Selected Features",
                    xaxis_title="Feature Selection Method",
                    yaxis_title="Number of Selected Features",
                )
                fig.show()
                barchart_counter += 1


base_path = "../../results"
list_of_experiments = [
    # "colon00",
    # "colon_s10",
    # "colon_s20",
    # "colon_s30",
    # "colon_s40",
    "colon_shift_00",
]
data_name = "colon"
# list_of_experiments = [
#     "prostate_shift_00",
#     "prostate_shift_01",
#     "prostate_shift_02",
# ]
# data_name = "prostate"
# list_of_experiments = [
#     "leukemia_shift_00",
# ]
# data_name = "leukemia_big"

# result_dict_path = Path("../../results/random_noise_lognormal_30_7000_01_result_dict.pkl")
# list_of_experiments = ["random_noise_lognormal_30_7000_01"]
# input_df = pd.read_csv("../../data/random_noise_lognormal_30_7000_01_data_df.csv")
# hold_out_df = None
# assert input_df.columns[0] == "label", input_df.columns[0]

# data_df.drop(columns=[data_df.columns[0]], inplace=True)
# input_data_df.to_csv("../../results/random_noise_lognormal_30_7000_01_data_df.csv", index=False)

# load the raw experiment results
# with open(Path("../../results/colon_shift_00_result_dict.pkl"), "rb") as file:
#     result_dict = pickle.load(file)

hold_out_df, input_df = load_train_test_data_for_standardized_sample_size(data_name)
plot_average_results(
    evaluate_repeated_experiments(input_df, hold_out_df, base_path, list_of_experiments),
    save=False,
    data_display_name=data_name.capitalize(),
    hold_out=True,
)

