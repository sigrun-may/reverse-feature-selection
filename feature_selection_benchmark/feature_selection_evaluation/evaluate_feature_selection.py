# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Evaluate the feature selection results for the given experiments."""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_selection_benchmark.data_loader_tools import load_train_test_data_for_standardized_sample_size
from feature_selection_benchmark.feature_selection_evaluation import stability_estimator


def calculate_performance(y_predict, y_predict_proba, y_true) -> dict[str, float]:
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
    performance_metrics_dict = {
        # for the given hold out test set the healthy control 0 is the minority class
        # "F1": f1_score(y_true, y_predict, average="binary", pos_label=0),
        "F1": f1_score(y_true, y_predict, average="weighted"),
        "AUPRC": average_precision_score(y_true, y_predict, average="weighted"),
        "AUROC": roc_auc_score(y_true, probability_positive_class, average="weighted"),
        "Precision": precision_score(y_true, y_predict, average="weighted"),
        "Recall": recall_score(y_true, y_predict, average="weighted"),
        # "F1": f1_score(y_true, y_predict, average="macro"),
        # "AUPRC": average_precision_score(y_true, y_predict),
        # "AUROC": roc_auc_score(y_true, probability_positive_class),
    }
    return performance_metrics_dict


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_importance: np.ndarray,
    seed: int,
) -> tuple:
    """Trains a model and predicts the labels for the test data.

    Args:
        train_data: The training data.
        test_data: The test data.
        feature_importance: The feature importance.
        seed: The random state for reproducibility.

    Returns:
        Prediction results: predicted labels, predicted probabilities, true labels.
    """
    # extract x and y values
    x_train = train_data.iloc[:, 1:].to_numpy()
    y_train = train_data.iloc[:, 0].to_numpy()
    x_test = test_data.iloc[:, 1:].to_numpy()
    y_test = test_data.iloc[:, 0].to_numpy()

    # -1 for label column
    assert train_data.shape[1] - 1 == feature_importance.size, train_data.shape[1]
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
        random_state=seed,  # Set random seed for reproducibility
    )
    clf.fit(feature_subset_x_train, y_train)
    predicted_proba_y = clf.predict_proba(feature_subset_x_test)
    predicted_y = clf.predict(feature_subset_x_test)
    return predicted_y, predicted_proba_y, y_test


def evaluate_reverse_random_forest(
    input_result_dict: dict,
    seed: int,
    experiment_data_df: pd.DataFrame,
    test_data_df: pd.DataFrame,
):
    """Evaluate the feature selection results of the reverse random forest method.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        seed: The random seed for reproducibility.
        experiment_data_df: The experiment data containing the features and labels.
        test_data_df: The hold-out data set as test data.
    """
    assert "reverse_random_forest" in input_result_dict
    loo_cv_iteration_list = input_result_dict["reverse_random_forest"]
    assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

    # initialize feature importance matrix with zeros
    feature_importance_matrix = np.zeros((len(loo_cv_iteration_list), len(loo_cv_iteration_list[0])))

    performance_metrics = []

    for i, cv_iteration_result in enumerate(loo_cv_iteration_list):
        feature_subset_selection_array = np.zeros(len(cv_iteration_result))

        # extract feature selection array
        for j, p_value in enumerate(cv_iteration_result["p_value"]):
            # check if the value of the "p_value" column is smaller or equal than 0.05
            if p_value is not None and p_value <= 0.05:
                # get index of the column with the fraction_mean value
                fraction_mean_index = cv_iteration_result.columns.get_loc("fraction_mean")

                # extract the fraction of selected features
                feature_subset_selection_array[j] = cv_iteration_result.iloc[j, fraction_mean_index]

        feature_importance_matrix[i, :] = feature_subset_selection_array

        # calculate the performance metrics
        train_data_df = experiment_data_df.drop(index=i)
        assert train_data_df.shape[0] == experiment_data_df.shape[0] - 1

        y_predict, y_predict_proba, y_true = train_and_predict(
            train_data_df, test_data_df, feature_subset_selection_array, seed
        )
        performance_metrics.append(calculate_performance(y_predict, y_predict_proba, y_true))

    # store the results in the result dictionary
    input_result_dict["evaluation"]["reverse_random_forest"] = {
        "importance_matrix": feature_importance_matrix,
        "stability": stability_estimator.calculate_stability(feature_importance_matrix),
        "number of selected features": np.count_nonzero(feature_importance_matrix, axis=1),
        "performance_metrics": pd.DataFrame(performance_metrics),
    }


def evaluate_standard_random_forest(
    input_result_dict: dict,
    seed: int,
    experiment_data_df: pd.DataFrame,
    test_data_df: pd.DataFrame = None,
):
    """Evaluate the feature selection results of the different standard random forest methods.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        seed: The random seed for reproducibility.
        experiment_data_df: The experiment data containing the features and labels.
        test_data_df: The hold-out data set as test data.
    """
    # evaluate the feature selection results for the standard random forest
    loo_cv_iteration_list = input_result_dict["standard_random_forest"]
    assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

    # evaluate each importance calculation method
    for importance_calculation_method in loo_cv_iteration_list[0]:
        # exclude the train oob score and the best hyperparameters
        if "score" in importance_calculation_method or "best" in importance_calculation_method:
            continue

        # initialize feature importance matrix with zeros
        feature_importance_matrix = np.zeros(
            (len(loo_cv_iteration_list), len(loo_cv_iteration_list[0][importance_calculation_method]))
        )

        performance_metrics = []

        # iterate over the leave-one-out cross-validation iterations
        for i, loo_cv_iteration in enumerate(loo_cv_iteration_list):
            feature_importances = loo_cv_iteration[importance_calculation_method]
            feature_importance_matrix[i] = feature_importances

            # calculate the performance metrics
            train_data_df = experiment_data_df.drop(index=i)
            assert train_data_df.shape[0] == experiment_data_df.shape[0] - 1
            y_predict, y_predict_proba, y_true = train_and_predict(
                train_data_df, test_data_df, feature_importances, seed
            )
            performance_metrics.append(calculate_performance(y_predict, y_predict_proba, y_true))

        # store the results in the result dictionary
        input_result_dict["evaluation"][importance_calculation_method] = {
            "importance_matrix": feature_importance_matrix,
            "stability": stability_estimator.calculate_stability(feature_importance_matrix),
            "number of selected features": np.count_nonzero(feature_importance_matrix, axis=1),
            "performance_metrics": pd.DataFrame(performance_metrics),
        }


def evaluate_experiment(
    input_result_dict: dict,
    seed: int,
    experiment_data_df: pd.DataFrame,
    test_data_df: pd.DataFrame = None,
):
    """Evaluate the feature selection results of the given experiment.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        seed: The random seed for reproducibility.
        experiment_data_df: The experiment data containing the features and labels.
        test_data_df: The test data to evaluate if a hold-out data set is available.
    """
    assert "standard_random_forest" in input_result_dict
    assert "standard_random_forest_meta_data" in input_result_dict
    assert "reverse_random_forest" in input_result_dict
    assert "reverse_random_forest_meta_data" in input_result_dict

    # initialize the evaluation results
    input_result_dict["evaluation"] = {}

    # evaluate the feature selection results for the standard random forest
    evaluate_standard_random_forest(input_result_dict, seed, experiment_data_df, test_data_df)
    evaluate_reverse_random_forest(input_result_dict, seed, experiment_data_df, test_data_df)

    # check if all evaluation performance metrics are different
    for performance_metric in input_result_dict["evaluation"]["reverse_random_forest"]["performance_metrics"]:
        comparison_series = input_result_dict["evaluation"]["reverse_random_forest"]["performance_metrics"][
            performance_metric
        ]
        for key, value in input_result_dict["evaluation"].items():
            if "reverse" not in key:
                # compare two columns of a pandas dataframe
                assert not comparison_series.equals(value["performance_metrics"][performance_metric])


def load_experiment_results(result_dict_path: str, experiment_id: str) -> dict:
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
    experiment_data_df: pd.DataFrame,
    test_data_df: pd.DataFrame,
    result_dict_path: str,
    repeated_experiments: list[str],
    seeds: list[int],
):
    """Evaluate the feature selection results and their reproducibility for each provided feature selection method.

    Args:
        experiment_data_df: The underlying original data for the given experiments.
        test_data_df: The test data to evaluate.
        result_dict_path: The base path to the pickled results files.
        repeated_experiments: A list of the experiment ids of repeated experiments to evaluate.
        seeds: A list of random seeds for the random forest classifier to ensure reproducibility.

    Returns:
        Evaluation results for each feature selection method.
    """
    random_states_rf = []
    random_seeds_reverse_rf = []

    repeated_result_dicts = []

    # iterate over all repeated experiments
    for experiment_id, seed in zip(repeated_experiments, seeds, strict=True):
        logging.info(f"Evaluating experiment {experiment_id}")
        result_dict = load_experiment_results(result_dict_path, experiment_id)
        random_states_rf.append(result_dict["standard_random_forest_meta_data"]["random_state"])
        if "reverse_random_forest_meta_data" in result_dict:
            random_seeds_reverse_rf.append(result_dict["reverse_random_forest_meta_data"]["random_seeds"])
        else:
            print(experiment_id)

        # check if the result dictionary already contains the evaluation results
        if "evaluation" in result_dict:
            logging.info(f"Evaluation results for {experiment_id} already present. Deleting them.")
            # delete the evaluation results to reevaluate the experiment
            del result_dict["evaluation"]

        # evaluate the experiment
        evaluate_experiment(result_dict, seed, experiment_data_df, test_data_df)
        repeated_result_dicts.append(result_dict)

    assert len(random_states_rf) == 3, "Random states are missing."
    assert len(set(random_states_rf)) == 3, "Random states are equal."

    if len(random_seeds_reverse_rf) != 3:
        print("Random seeds are missing. Only found " + str(len(random_seeds_reverse_rf)))

    # flatten list of seeds to check if the random seeds are equal
    flattened_lists_of_seeds: list[int] = sum(random_seeds_reverse_rf, [])
    if len(set(flattened_lists_of_seeds)) != 90:
        # find equal elements within the list
        equal_elements = [
            element for element in flattened_lists_of_seeds if flattened_lists_of_seeds.count(element) > 1
        ]
        print("total number of equal elements: ", len(equal_elements))
        print(set(equal_elements))
        print("number of equal elements: ", len(set(equal_elements)))

    return repeated_result_dicts


def summarize_evaluation_results(
    list_of_result_dicts: list[dict],
) -> dict:
    """Summarize the evaluation results for the given list of result dictionaries.

    Args:
        list_of_result_dicts: A list of result dictionaries containing the evaluation results.

    Returns:
        The summarized evaluation results.
    """
    assert len(list_of_result_dicts) == 3, len(list_of_result_dicts)
    summarized_result_dict = {}
    for result_dict in list_of_result_dicts:
        for key, value in result_dict["evaluation"].items():
            # insert number of selected features into the performance metrics dataframe
            value["performance_metrics"]["number of selected features"] = value["number of selected features"]

            # insert stability into the performance metrics dataframe
            value["performance_metrics"]["stability"] = value["stability"]

            if key not in summarized_result_dict:
                summarized_result_dict[key] = value["performance_metrics"]
            else:
                summarized_result_dict[key] = pd.concat(
                    [summarized_result_dict[key], value["performance_metrics"]], axis=0
                )

    summarized_df = pd.DataFrame(columns=summarized_result_dict["reverse_random_forest"].columns)
    for key in summarized_result_dict:
        assert summarized_result_dict[key].shape[0] == 90, summarized_result_dict[key].shape[0]

        # add row to the summarized dataframe with the mean values
        summarized_df.loc[key] = summarized_result_dict[key].mean()

        # add row to the summarized dataframe with the standard deviation values
        summarized_df.loc[f"{key}_std"] = summarized_result_dict[key].std()
    return summarized_result_dict


def plot_average_results(
    summarized_result_dict: dict,
    save: bool = False,
    path: str = "",
    data_display_name: str = "Data",
):
    """Plot the average results and the standard deviation for the given result dictionary.

    Args:
        summarized_result_dict: The dictionary containing the evaluated results to plot.
        save: A boolean indicating whether to save the plot as a pdf. Defaults to False.
        path: The path to save the plot. Defaults to an empty string.
        data_display_name: The name of the data to plot. Defaults to "Data".
    """
    for performance_metric in summarized_result_dict["reverse_random_forest"]:
        if performance_metric in ["number of selected features", "stability"]:
            continue
        # Create a DataFrame
        plot_df = pd.DataFrame(
            columns=[
                "method",
                "performance",
                "stability",
                "e_performance",
                "e_stability",
            ]
        )
        for feature_selection_method in summarized_result_dict:
            # print("feature_selection_method", feature_selection_method)
            plot_df.loc[len(plot_df)] = [
                feature_selection_method,
                np.mean(summarized_result_dict[feature_selection_method][performance_metric]),
                np.mean(summarized_result_dict[feature_selection_method]["stability"]),
                np.std(summarized_result_dict[feature_selection_method][performance_metric]),
                np.std(summarized_result_dict[feature_selection_method]["stability"]),
            ]
        # # plot the stability of the feature selection methods
        # fig_stability = px.bar(
        #     df,
        #     x="method",
        #     y="stability",
        #     error_y="e_stability",
        #     text="stability",
        #     template="plotly_white",
        #     height=350,
        # )
        # fig_stability.update_traces(textposition="outside")
        # fig_stability.update_layout(
        #     title=f"{data_display_name}: Stability of Feature Selection",
        #     xaxis_title="Feature Selection Method",
        #     yaxis_title="Stability",
        # )
        # # save figure as pdf
        # if save:
        #     plot_path = f"{path}/{data_display_name}_stability.pdf"
        #     pio.write_image(fig_stability, plot_path)
        # fig_stability.show()

        fig = px.scatter(
            plot_df,
            x="stability",
            y="performance",
            error_y="e_performance",
            error_x="e_stability",
            hover_data=["method"],
            text=[
                "gini",
                "without HPO",
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
            plot_path = f"{path}/{data_display_name}_{performance_metric}.pdf"
            pio.write_image(fig, plot_path)
        fig.show()


def evaluate_data(data_id: str, base_path: str, list_of_experiments: list[str], seeds: list[int]):
    """Evaluate the feature selection results for the given data.

    Args:
        data_id: The id of the data set to evaluate.
        base_path: The base path to the pickled results files.
        list_of_experiments: A list of the experiment ids of repeated experiments to evaluate.
        seeds: A list of random seeds for the random forest classifier to ensure reproducibility.

    Returns:
        The summarized evaluation results.
    """
    hold_out_df, input_df = load_train_test_data_for_standardized_sample_size(data_id)

    # evaluate the feature selection results for the given data
    list_of_result_dicts = evaluate_repeated_experiments(input_df, hold_out_df, base_path, list_of_experiments, seeds)
    print(data_id)
    summarized_evaluation_results = summarize_evaluation_results(list_of_result_dicts)
    # plot_average_results(
    #     summarized_evaluation_results,
    #     save=True,
    #     path=f"{base_path}/images",
    #     data_display_name=data_name,
    #     hold_out=True,
    #     loo=False,
    # )

    return summarized_evaluation_results


def create_latex_table(summarized_evaluation_results: dict, data_display_name: str):
    """Create a latex table for the given summarized evaluation results.

    Args:
        summarized_evaluation_results: The summarized evaluation results.
        data_display_name: The name of the data set to display.
    """
    df_for_latex = pd.DataFrame()

    for feature_selection_method, df in summarized_evaluation_results.items():
        feature_selection_method = feature_selection_method.replace("_", " ")

        # calculate the mean and standard deviation for each column
        std_df = df.std().to_frame().T
        mean_df = df.mean().to_frame().T
        mean_df.index = [feature_selection_method]
        for column in mean_df.columns:
            # round the mean values to five decimal places
            mean_df[column] = mean_df[column].apply(lambda x: f"{x:.5f}")

            # round the standard deviation values to five decimal places and insert latex code
            std_df[column] = std_df[column].apply(lambda x: f"$\pm {x:.5f}$")

            # merge standard deviation and mean values
            for mean, std in zip(mean_df[column], std_df[column], strict=True):
                mean_df.loc[feature_selection_method, column] = mean + std

        # insert row into the df_for_latex dataframe
        df_for_latex = pd.concat([df_for_latex, mean_df], axis=0)

    # print the dataframe as latex table
    for i in range(3):
        print(
            df_for_latex.iloc[:, (i * 3) : (i * 3) + 3].to_latex(
                index=True, caption=f"Performance metrics for {data_display_name}"
            )
        )
    print(df_for_latex.to_latex(index=True, caption=f"Performance metrics for {data_display_name}"))


def run_evaluation():
    """Run the evaluation for the given experiments."""
    # seeds for the random forest classifier for each repeated experiment
    seeds = [2042, 2043, 2044]

    # base path to the pickled results files
    base_path = "../../results"

    # path to directory to save the plots
    base_path_to_save_plots = "../../results/images"

    experiment_id_list = [
        "colon_s10",
        "colon_s20",
        # "colon_s30",
        # "colon_s40",
        "colon_shift_00",
    ]
    # experiment_id_list = [
    #     "colon_0",
    #     "colon_1",
    #     "colon_2",
    # ]
    data_id = "colon"
    evaluation_result_colon = evaluate_data(data_id, base_path, experiment_id_list, seeds)
    create_latex_table(evaluation_result_colon, "Colon Cancer")
    plot_average_results(
        evaluation_result_colon, save=False, path=base_path_to_save_plots, data_display_name="Colon Cancer"
    )

    experiment_id_list = [
        "prostate_shift_00",
        "prostate_shift_01",
        "prostate_shift_02",
    ]
    # experiment_id_list = [
    #     "prostate_0",
    #     "prostate_1",
    #     "prostate_2",
    # ]
    data_id = "prostate"
    evaluation_result_prostate = evaluate_data(data_id, base_path, experiment_id_list, seeds)
    create_latex_table(evaluation_result_prostate, "Prostate Cancer")
    plot_average_results(
        evaluation_result_prostate, save=False, path=base_path_to_save_plots, data_display_name="Prostate Cancer"
    )

    experiment_id_list = [
        "leukemia_shift_00",
        "leukemia_big_shift10",
        "leukemia_shift_02",
    ]
    # experiment_id_list = [
    #     "leukemia_big_0",
    #     "leukemia_big_1",
    #     "leukemia_big_2",
    # ]
    data_id = "leukemia_big"
    evaluation_result_leukemia = evaluate_data(data_id, base_path, experiment_id_list, seeds)
    create_latex_table(evaluation_result_leukemia, "Leukemia")
    plot_average_results(
        evaluation_result_leukemia, save=False, path=base_path_to_save_plots, data_display_name="Leukemia"
    )


run_evaluation()
