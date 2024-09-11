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

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

from feature_selection_benchmark.data_loader_tools import (
    load_train_test_data_for_standardized_sample_size,
    standardize_sample_size_of_hold_out_data_single_df,
)
from feature_selection_benchmark.feature_selection_evaluation import stability_estimator

pio.kaleido.scope.mathjax = None


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
        "Average Precision Score": average_precision_score(y_true, y_predict),
        "AUC": roc_auc_score(y_true, probability_positive_class),
        "Accuracy": accuracy_score(y_true, y_predict),
        # "Specificity": recall_score(y_true, y_predict, pos_label=0),  # recall for the negative class
        # "Sensitivity": recall_score(y_true, y_predict, pos_label=1),  # recall for the positive class
        # "F1": f1_score(y_true, y_predict, average="macro"),
        # "AUPRC": average_precision_score(y_true, y_predict),
        # "AUROC": roc_auc_score(y_true, probability_positive_class),
    }
    return performance_metrics_dict


def calculate_performance_metrics_on_hold_out_data(
    feature_subset_selection, shuffle_seed_hold_out_data, hold_out_test_data_df, seed_for_random_forest, train_data_df
):
    """Calculate the performance metrics for the given hold-out test data.

    Args:
        feature_subset_selection: The selected feature subset.
        shuffle_seed_hold_out_data: The random seed for shuffling the hold-out test data.
        hold_out_test_data_df: The hold-out test data.
        seed_for_random_forest: The random seed for reproducibility of the random forest.
        train_data_df: The training data.

    Returns:
        The performance metrics for the given hold-out test data.
    """
    standardized_hold_out_test_data_df = standardize_sample_size_of_hold_out_data_single_df(
        hold_out_test_data_df, shuffle_seed=shuffle_seed_hold_out_data
    )
    y_predict, y_predict_proba, y_true = train_and_predict(
        train_data_df,
        standardized_hold_out_test_data_df,
        feature_subset_selection,
        seed_for_random_forest,
    )
    return calculate_performance(y_predict, y_predict_proba, y_true)


def calculate_performance_metrics_on_shuffled_hold_out_subset(
    train_data_df, hold_out_test_data_df, feature_subset_selection, seed_for_random_forest
) -> list[dict[str, float]]:
    """Calculate the performance metrics for the given hold-out test data.

    Args:
        train_data_df: The training data.
        hold_out_test_data_df: The hold-out test data.
        feature_subset_selection: The selected feature subset.
        seed_for_random_forest: The random seed for reproducibility of the random forest.

    Returns:
        The performance metrics for the given hold-out test data.
    """
    # parallelize iterations of shuffling the hold out test data with joblib
    return joblib.Parallel(n_jobs=4)(
        joblib.delayed(calculate_performance_metrics_on_hold_out_data)(
            feature_subset_selection,
            hold_out_shuffle_iteration,
            hold_out_test_data_df,
            seed_for_random_forest,
            train_data_df,
        )
        for hold_out_shuffle_iteration in range(16)
    )


def evaluate_reverse_random_forest(
    input_result_dict: dict,
    seed_for_random_forest: int,
    experiment_data_df: pd.DataFrame,
    hold_out_test_data_df: pd.DataFrame,
):
    """Evaluate the feature selection results of the reverse random forest method.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        seed_for_random_forest: The random seed for reproducibility.
        experiment_data_df: The experiment data containing the features and labels.
        hold_out_test_data_df: The hold-out data set as test data.
    """
    assert "reverse_random_forest" in input_result_dict
    loo_cv_iteration_list = input_result_dict["reverse_random_forest"]
    assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

    # initialize the flip parameter for the stability estimator
    flip = False

    # initialize feature importance matrix with zeros
    feature_importance_matrix = np.zeros((len(loo_cv_iteration_list), len(loo_cv_iteration_list[0])))
    performance_metrics = []

    # iterate over the leave-one-out cross-validation iterations
    for i, cv_iteration_result in enumerate(loo_cv_iteration_list):
        # if "feature_subset_selection" not in cv_iteration_result:
        #     feature_subset_selection_array = np.zeros(len(cv_iteration_result))
        #
        #     # extract feature selection array
        #     for j, p_value in enumerate(cv_iteration_result["p_value"]):
        #         # check if the value of the "p_value" column is smaller or equal than 0.05
        #         if p_value is not None and p_value <= 0.05:
        #             # get index of the column with the fraction_mean value
        #             fraction_mean_index = cv_iteration_result.columns.get_loc("fraction_mean")
        #
        #             # extract the fraction of selected features
        #             feature_subset_selection_array[j] = cv_iteration_result.iloc[j, fraction_mean_index]
        #
        #     cv_iteration_result["feature_subset_selection"] = feature_subset_selection_array

        feature_importance_matrix[i] = cv_iteration_result["feature_subset_selection"]

        # check if the feature subset selection is empty
        if np.count_nonzero(cv_iteration_result["feature_subset_selection"]) == 0:
            flip = True  # flip the feature importance matrix for the stability estimator to avoid division by zero
            continue  # skip calculating the performance metrics

        # calculate the performance metrics
        train_data_df = experiment_data_df.drop(index=i)
        assert train_data_df.shape[0] == experiment_data_df.shape[0] - 1
        performance_metrics.extend(
            calculate_performance_metrics_on_shuffled_hold_out_subset(
                train_data_df,
                hold_out_test_data_df,
                cv_iteration_result["feature_subset_selection"],
                seed_for_random_forest,
            )
        )

    # store the results in the result dictionary
    input_result_dict["evaluation"]["reverse_random_forest"] = {
        "importance_matrix": feature_importance_matrix,
        "stability": stability_estimator.calculate_stability(feature_importance_matrix, flip),
        "number of selected features": np.count_nonzero(feature_importance_matrix, axis=1),
        "performance_metrics": pd.DataFrame(performance_metrics),
    }


def evaluate_standard_random_forest(
    input_result_dict: dict,
    seed_random_forest: int,
    experiment_data_df: pd.DataFrame,
    hold_out_test_data_df: pd.DataFrame,
):
    """Evaluate the feature selection results of the different standard random forest methods.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        seed_random_forest: The random seed for reproducibility of the random forest.
        experiment_data_df: The experiment data containing the features and labels.
        hold_out_test_data_df: The hold-out data set as test data.
    """
    # evaluate the feature selection results for the standard random forest
    loo_cv_iteration_list = input_result_dict["standard_random_forest"]
    assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

    # initialize the flip parameter for the stability estimator
    flip = False

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

            if np.count_nonzero(feature_importances) == 0:
                flip = True  # flip the feature importance matrix for the stability estimator to avoid division by zero
                continue  # skip calculating the performance metrics

            # calculate the performance metrics
            train_data_df = experiment_data_df.drop(index=i)
            assert train_data_df.shape[0] == experiment_data_df.shape[0] - 1

            # parallelize 8 iterations of shuffling the hold out test data with joblib
            performance_metrics.extend(
                joblib.Parallel(n_jobs=4)(
                    joblib.delayed(calculate_performance_metrics_on_hold_out_data)(
                        feature_importances,
                        hold_out_shuffle_iteration,
                        hold_out_test_data_df,
                        seed_random_forest,
                        train_data_df,
                    )
                    for hold_out_shuffle_iteration in range(16)
                )
            )
            # y_predict, y_predict_proba, y_true = train_and_predict(
            #     train_data_df, hold_out_test_data_df, feature_importances, seed_random_forest
            # )
            # performance_metrics.append(calculate_performance(y_predict, y_predict_proba, y_true))
        #
        # binarize the feature importance matrix
        feature_importance_matrix = np.where(feature_importance_matrix != 0, 1, 0)

        # store the results in the result dictionary
        input_result_dict["evaluation"][importance_calculation_method] = {
            "importance_matrix": feature_importance_matrix,
            "stability": stability_estimator.calculate_stability(feature_importance_matrix, flip),
            # "stability": 0.0,
            "number of selected features": np.count_nonzero(feature_importance_matrix, axis=1),
            "performance_metrics": pd.DataFrame(performance_metrics),
        }


def evaluate_experiment(
    input_result_dict: dict,
    seed_random_forest: int,
):
    """Evaluate the feature selection results of the given experiment.

    Args:
        input_result_dict: The result dictionary containing the feature selection results.
        seed_random_forest: The random seed for reproducibility of the random forest.
    """
    assert "standard_random_forest" in input_result_dict
    assert "standard_random_forest_meta_data" in input_result_dict
    assert "reverse_random_forest" in input_result_dict
    assert "reverse_random_forest_meta_data" in input_result_dict

    # initialize the evaluation results
    input_result_dict["evaluation"] = {}

    # evaluate the feature selection results for the standard random forest
    experiment_data_df, hold_out_test_data_df = load_train_test_data_for_standardized_sample_size(
        input_result_dict["reverse_random_forest_meta_data"]
    )
    evaluate_standard_random_forest(input_result_dict, seed_random_forest, experiment_data_df, hold_out_test_data_df)
    evaluate_reverse_random_forest(input_result_dict, seed_random_forest, experiment_data_df, hold_out_test_data_df)

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
    result_dict_path: str,
    repeated_experiments: list[str],
    seeds: list[int],
):
    """Evaluate the feature selection results and their reproducibility for each provided feature selection method.

    Args:
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
        evaluate_experiment(result_dict, seed)
        repeated_result_dicts.append(result_dict)

    assert len(random_states_rf) == len(repeated_experiments), "Random states are missing."
    assert len(set(random_states_rf)) == len(repeated_experiments), "Random states are equal."

    if len(random_seeds_reverse_rf) != len(repeated_experiments):
        print("Random seeds are missing. Only found " + str(len(random_seeds_reverse_rf)))

    # flatten list of seeds to check if the random seeds are equal
    flattened_lists_of_seeds: list[int] = sum(random_seeds_reverse_rf, [])
    if len(set(flattened_lists_of_seeds)) != len(repeated_experiments) * 30:
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
) -> pd.DataFrame:
    """Summarize the evaluation results for the given list of result dictionaries.

    Args:
        list_of_result_dicts: A list of result dictionaries containing the evaluation results.

    Returns:
        The summarized evaluation results.
    """
    summarized_result_dict: dict[str, dict] = {}
    for result_dict in list_of_result_dicts:
        for key, value in result_dict["evaluation"].items():
            assert isinstance(value["stability"], float)
            if key not in summarized_result_dict:
                summarized_result_dict[key] = {
                    "performance_metrics": value["performance_metrics"],
                    "stability": [value["stability"]],
                    "number of selected features": value["number of selected features"].tolist(),
                }
            else:
                summarized_result_dict[key]["performance_metrics"] = pd.concat(
                    [summarized_result_dict[key]["performance_metrics"], value["performance_metrics"]], axis=0
                )
                summarized_result_dict[key]["stability"].append(value["stability"])
                summarized_result_dict[key]["number of selected features"].extend(value["number of selected features"])

    # initialize the summarized dataframe with the columns of the performance metrics
    summarized_df = pd.DataFrame(columns=summarized_result_dict["reverse_random_forest"]["performance_metrics"].columns)
    for feature_selection_method_name, metrics in summarized_result_dict.items():
        if "shap" in feature_selection_method_name:
            continue

        summarized_df.loc[feature_selection_method_name] = metrics["performance_metrics"].mean()
        summarized_df.loc[f"{feature_selection_method_name}_std"] = metrics["performance_metrics"].std()
        summarized_df.loc[feature_selection_method_name, "stability"] = np.mean(metrics["stability"])
        summarized_df.loc[f"{feature_selection_method_name}_std", "stability"] = np.std(metrics["stability"])
        summarized_df.loc[feature_selection_method_name, "number of selected features"] = np.mean(
            metrics["number of selected features"]
        )
        summarized_df.loc[f"{feature_selection_method_name}_std", "number of selected features"] = np.std(
            metrics["number of selected features"]
        )

    return summarized_df


def create_latex_table(summarized_evaluation_results_df: pd.DataFrame, data_display_name: str):
    """Create a latex table for the given summarized evaluation results.

    Args:
        summarized_evaluation_results_df: The summarized evaluation results.
        data_display_name: The name of the data set to display.
    """
    df_for_latex = summarized_evaluation_results_df.copy()

    for column_name in df_for_latex:
        # round the floats to five decimal places
        df_for_latex[column_name] = df_for_latex[column_name].apply(lambda x: f"{x:.5f}")

        for index_name in df_for_latex.index:
            if "std" not in index_name:
                # insert latex code and append std to mean values
                df_for_latex.loc[index_name, column_name] = (
                    f"${df_for_latex.loc[index_name, column_name]}"
                    f"\pm{df_for_latex.loc[index_name+'_std', column_name]}$"
                )
            else:
                continue

    # remove rows with "_std" from the index
    df_for_latex = df_for_latex.drop(index=[index for index in df_for_latex.index if "std" in index])

    np.testing.assert_array_equal(
        np.asarray(summarized_evaluation_results_df.columns), np.asarray(df_for_latex.columns)
    )
    assert int(len(summarized_evaluation_results_df.index) / 2) == len(df_for_latex.index), (
        str(len(df_for_latex.index)) + " " + str(len(summarized_evaluation_results_df.index))
    )

    # remove underscores from index
    df_for_latex.index = [feature_selection_method.replace("_", " ") for feature_selection_method in df_for_latex.index]

    # print the dataframe as latex table
    performance_metrics_list = []

    for metric in df_for_latex.columns:
        if metric in ["number of selected features", "stability"]:
            continue
        performance_metrics_list.append(metric)

    if len(performance_metrics_list) > 2:
        counter = len(performance_metrics_list)
        for i in range(int(counter / 3)):
            print(
                df_for_latex[performance_metrics_list[i * 3 : (i + 1) * 3]].to_latex(
                    index=True, caption=f"Performance metrics for {data_display_name}"
                )
            )
    else:
        print(
            df_for_latex[performance_metrics_list].to_latex(
                index=True, caption=f"Performance metrics for {data_display_name}"
            )
        )
    print(
        df_for_latex[["number of selected features", "stability"]].to_latex(
            index=True, caption=f"Number of selected features and stability for {data_display_name}"
        )
    )


def plot_average_results(
    summarized_result_df: pd.DataFrame,
    save: bool = False,
    path: str = "",
    data_display_name: str = "Data",
):
    """Plot the average results and the standard deviation for the given result dictionary.

    Args:
        summarized_result_df: The dataframe containing the evaluated results to plot.
        save: A boolean indicating whether to save the plot as a pdf. Defaults to False.
        path: The path to save the plot. Defaults to an empty string.
        data_display_name: The name of the data to plot. Defaults to "Data".
    """
    # extract the standard deviation values for each performance metric
    column_lists = [
        [summarized_result_df.loc[idx, column_name] for idx in summarized_result_df.index if "std" in idx]
        for column_name in summarized_result_df.columns
    ]

    # delete rows with "_std" from the index
    summarized_result_df = summarized_result_df.drop(
        index=[index for index in summarized_result_df.index if "std" in index]
    )

    # insert std as columns with column_name_std to the dataframe
    for std_list, column in zip(column_lists, summarized_result_df.columns, strict=True):
        summarized_result_df.loc[:, f"{column}_std"] = std_list

    summarized_result_df["method"] = summarized_result_df.index

    x_axis_title = "Stability of Feature Selection"
    for performance_metric in summarized_result_df.columns:
        if performance_metric in ["number of selected features", "stability", "method"] or "std" in performance_metric:
            continue

        fig = px.scatter(
            summarized_result_df,
            x="stability",
            y=performance_metric,
            error_y=f"{performance_metric}_std",
            error_x="stability_std",
            hover_data=["method"],
            text=[
                "gini",
                "without HPO",
                # "shap",
                "gini corrected",
                "permutation",
                "reverse random forest",
            ],  # "method",  # Add method name to each data point
            template="plotly_white",
            height=250,
            color="method",
        )
        # unoptimized, gini, shap, permutation, reverse
        if data_display_name == "Leukemia":
            fig.update_traces(textposition=["top left", "bottom right", "bottom center", "top right", "bottom left"])
        elif "olon" in data_display_name:
            fig.update_traces(textposition=["top left", "bottom right", "bottom center", "top left", "bottom right"])
        else:
            fig.update_traces(textposition=["top left", "top right", "bottom center", "top left", "bottom right"])
        fig.update_xaxes(range=[0, 1.01])
        fig.update_yaxes(range=[0, 1.09])
        fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=performance_metric,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            showlegend=False,
        )
        # save figure as pdf
        if save:
            plot_path = f"{path}/{data_display_name}_{performance_metric}.pdf"
            pio.write_image(fig, plot_path)
        fig.show()

        # remove x-axis title
        x_axis_title = ""


def evaluate_data(
    base_path: str, list_of_experiments: list[str], seeds: list[int], data_display_name: str, save_plots: bool = False
):
    """Evaluate the feature selection results for the given data.

    Args:
        base_path: The base path to the pickled results files.
        list_of_experiments: A list of the experiment ids of repeated experiments to evaluate.
        seeds: A list of random seeds for the random forest classifier to ensure reproducibility.
        data_display_name: The name of the data set to display.
        save_plots: A boolean indicating whether to save the plots. Defaults to False.

    Returns:
        The summarized evaluation results.
    """
    # evaluate the feature selection results for the given data
    list_of_result_dicts = evaluate_repeated_experiments(base_path, list_of_experiments, seeds)
    summarized_evaluation_results = summarize_evaluation_results(list_of_result_dicts)
    create_latex_table(summarized_evaluation_results, data_display_name)

    # plot the average results and the standard deviation
    if save_plots:
        # path to directory to save the plots
        base_path_to_save_plots = f"{base_path}/images"
        os.makedirs(base_path_to_save_plots, exist_ok=True)
    else:
        base_path_to_save_plots = ""
    plot_average_results(summarized_evaluation_results, save_plots, base_path_to_save_plots, data_display_name)
    return summarized_evaluation_results


def run_evaluation():
    """Run the evaluation for the given experiments."""
    # seeds for the random forest classifier for each repeated experiment
    seeds = [2042, 2043, 2044]
    # seeds = [2042, 2043]

    # base path to the pickled results files
    base_path = "../../results"

    # evaluate the feature selection results for the given data
    experiments_dict = {
        # "Colon Cancer": [
        #     "colon_0_shuffle_seed_None",
        #     "colon_1_shuffle_seed_None",
        #     "colon_2_shuffle_seed_None",
        # ],
        # "Prostate Cancer": [
        #     "prostate_0_shuffle_seed_None",
        #     "prostate_1_shuffle_seed_None",
        #     "prostate_2_shuffle_seed_None",
        # ],
        # "Leukemia": [
        #     "leukemia_big_0_shuffle_seed_None",
        #     "leukemia_big_1_shuffle_seed_None",
        #     "leukemia_big_2_shuffle_seed_None",
        # ],
        "Colon Cancer": [
            "colon_0_shuffle_seed_None_pcg",
            "colon_1_shuffle_seed_None_pcg",
            "colon_2_shuffle_seed_None_pcg",
        ],
        "Prostate Cancer": [
            "prostate_0_shuffle_seed_None_pcg",
            "prostate_1_shuffle_seed_None_pcg",
            "prostate_2_shuffle_seed_None_pcg",
        ],
        "Leukemia": [
            "leukemia_big_0_shuffle_seed_None_pcg",
            "leukemia_big_1_shuffle_seed_None_pcg",
            "leukemia_big_2_shuffle_seed_None_pcg",
        ],
        # "Random Noise Lognormal": [
        #     "random_noise_lognormal_0_shuffle_seed_None_srf",
        #     "random_noise_lognormal_1_shuffle_seed_None_srf",
        #     "random_noise_lognormal_2_shuffle_seed_None_srf",
        # ],
        # "Random Noise Normal": [
        #     "random_noise_normal_0_shuffle_seed_None_srf",
        #     "random_noise_normal_1_shuffle_seed_None_srf",
        #     "random_noise_normal_2_shuffle_seed_None_srf",
        # ],
        # "Colon Cancer": [
        #     "colon_s10",
        #     "colon_s20",
        #     # "colon_s30",
        #     # "colon_s40",
        #     "colon_shift_00",
        # ],
        # "Prostate Cancer": [
        #     "prostate_shift_00",
        #     "prostate_shift_01",
        #     "prostate_shift_02",
        # ],
        # "Leukemia": [
        #     "leukemia_shift_00",
        #     "leukemia_big_shift10",
        #     "leukemia_shift_02",
        # ],
    }
    for data_display_name, experiment_id_list in experiments_dict.items():
        evaluate_data(base_path, experiment_id_list, seeds, data_display_name=data_display_name, save_plots=True)


run_evaluation()
