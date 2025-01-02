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
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_selection_benchmark.data_loader_tools import (
    load_train_test_data_for_standardized_sample_size,
    standardize_sample_size_of_hold_out_data_single_df,
)
from feature_selection_benchmark.feature_selection_evaluation import stability_estimator

pio.kaleido.scope.mathjax = None

# initialize the logger
logging.basicConfig(level=logging.INFO)

# define number of jobs for parallel processing
N_JOBS = 1


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_importance,
    seed: int,
) -> tuple:
    """Trains a model and predicts the labels for the test data.

    Args:
        train_data: The training data.
        test_data: The test data.
        feature_importance: The feature importance. Must be array_like for numpy.
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
        random_state=seed,  # Set random seed for reproducibility
    )
    clf.fit(feature_subset_x_train, y_train)
    predicted_proba_y = clf.predict_proba(feature_subset_x_test)
    predicted_y = clf.predict(feature_subset_x_test)
    return predicted_y, predicted_proba_y, y_test


def calculate_performance_metrics(
    train_data_df: pd.DataFrame,
    feature_subset_selection,
    hold_out_test_data_df: pd.DataFrame,
    shuffle_seed_hold_out_data: int,
    seed_for_random_forest: int,
):
    """Calculate the performance metrics for the given hold-out test data.

    Args:
        train_data_df: The training data.
        feature_subset_selection: The selected feature subset. Must be array_like for numpy.
        hold_out_test_data_df: The hold-out test data.
        shuffle_seed_hold_out_data: The random seed for shuffling the hold-out test data. Corresponds to the integer
            of the number of hold-out test data shuffles starting with zero.
        seed_for_random_forest: The random seed for reproducibility of the random forest.

    Returns:
        The performance metrics for the given hold-out test data.
    """
    assert np.count_nonzero(feature_subset_selection) > 0, "No features selected."
    assert "label" in train_data_df.columns, "Label column is missing."
    assert "label" in hold_out_test_data_df.columns, "Label column is missing."
    assert train_data_df.shape[1] - 1 == feature_subset_selection.size, train_data_df.shape[1]
    assert hold_out_test_data_df.shape[1] - 1 == feature_subset_selection.size, hold_out_test_data_df.shape[1]
    # check if train_data_df and hold_out_test_data_df are pd.DataFrames
    assert isinstance(train_data_df, pd.DataFrame), "train_data_df is not a pd.DataFrame."
    assert isinstance(hold_out_test_data_df, pd.DataFrame), "hold_out_test_data_df is not a pd.DataFrame."
    # check if the indices are unique
    assert train_data_df.index.is_unique, "Indices are not unique."
    assert hold_out_test_data_df.index.is_unique, "Indices are not unique."

    standardized_hold_out_test_data_df = standardize_sample_size_of_hold_out_data_single_df(
        hold_out_test_data_df, shuffle_seed=shuffle_seed_hold_out_data
    )
    y_predict, y_predict_proba, y_true = train_and_predict(
        train_data_df,
        standardized_hold_out_test_data_df,
        feature_subset_selection,
        seed_for_random_forest,
    )
    # extract the probability for the positive class for briar score calculation
    probability_positive_class = np.array([proba[1] for proba in y_predict_proba])
    performance_metrics_dict = {
        # for the given hold out test set the healthy control 0 is the minority class
        "Average Precision Score": average_precision_score(y_true, y_predict),
        "AUC": roc_auc_score(y_true, probability_positive_class),
        "Accuracy": accuracy_score(y_true, y_predict),
        "Sensitivity": recall_score(y_true, y_predict),
        "Specificity": recall_score(y_true, y_predict, pos_label=0),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_predict),
        "Precision": precision_score(y_true, y_predict),
    }
    return performance_metrics_dict


def calculate_performance_metrics_on_shuffled_hold_out_subset(
    train_data_df: pd.DataFrame,
    feature_subset_selection,
    hold_out_test_data_df: pd.DataFrame,
    seed_for_random_forest: int,
) -> list[dict[str, float]]:
    """Calculate the performance metrics for the given hold-out test data.

    Args:
        train_data_df: The training data.
        feature_subset_selection: The selected feature subset. Must be array_like for numpy.
        hold_out_test_data_df: The hold-out test data.
        seed_for_random_forest: The random seed for reproducibility of the random forest.

    Returns:
        The performance metrics for the given hold-out test data.
    """
    # parallelize iterations of shuffling the hold out test data with joblib
    return joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(calculate_performance_metrics)(
            train_data_df,
            feature_subset_selection,
            hold_out_test_data_df,
            (hold_out_shuffle_iteration+10000),
            seed_for_random_forest,
        )
        for hold_out_shuffle_iteration in range(1)
    )


def evaluate_feature_selection(
    loo_cv_iteration_list: list,
    experiment_data_df: pd.DataFrame,
    hold_out_test_data_df: pd.DataFrame,
    seed_for_random_forest: int,
    feature_selection_method_key: str,
) -> dict:
    """Evaluate the feature selection results.

    Args:
        loo_cv_iteration_list: List of leave-one-out cross-validation iterations.
        experiment_data_df: The experiment data containing the features and labels.
        hold_out_test_data_df: The hold-out data set as test data.
        seed_for_random_forest: The random seed for reproducibility.
        feature_selection_method_key: The key to access feature importance in the iteration results.

    Returns:
        A dictionary containing the evaluation results.
    """
    # flip is set to True if the feature subset selection is empty for at least one iteration
    flip = False
    feature_importance_matrix = np.zeros(
        (len(loo_cv_iteration_list), len(loo_cv_iteration_list[0][feature_selection_method_key]))
    )
    performance_metrics_list = []
    for loo_idx, cv_iteration_result in enumerate(loo_cv_iteration_list):
        feature_importances = cv_iteration_result[feature_selection_method_key]
        if np.count_nonzero(feature_importances) == 0:
            flip = True
            logging.warning(
                f"Feature subset selection is empty for iteration {loo_idx}, "
                f"skipping performance metrics calculation."
            )
            continue

        feature_importance_matrix[loo_idx] = feature_importances
        train_data_df = experiment_data_df.drop(index=loo_idx)
        assert train_data_df.shape[0] == experiment_data_df.shape[0] - 1

        performance_metrics_list.extend(
            calculate_performance_metrics_on_shuffled_hold_out_subset(
                train_data_df,
                feature_importances,
                hold_out_test_data_df,
                seed_for_random_forest,
            )
        )
    return {
        "importance_matrix": feature_importance_matrix,
        "stability": stability_estimator.calculate_stability(feature_importance_matrix, flip),
        "number of selected features": np.count_nonzero(feature_importance_matrix, axis=1),
        "performance_metrics_df": pd.DataFrame(performance_metrics_list),
    }


def evaluate_benchmark_experiment(
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

    # load the experiment data and the hold-out test data
    experiment_data_df, hold_out_test_data_df = load_train_test_data_for_standardized_sample_size(
        input_result_dict["reverse_random_forest_meta_data"]
    )
    # evaluate the feature selection results for the different standard random forest methods
    loo_cv_iteration_list = input_result_dict["standard_random_forest"]
    assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

    for importance_calculation_method in loo_cv_iteration_list[0]:
        if "score" in importance_calculation_method or "best" in importance_calculation_method:
            continue

        input_result_dict["evaluation"][importance_calculation_method] = evaluate_feature_selection(
            loo_cv_iteration_list,
            experiment_data_df,
            hold_out_test_data_df,
            seed_random_forest,
            importance_calculation_method,
        )

    # evaluate the feature selection results for the reverse random forest method
    loo_cv_iteration_list = input_result_dict["reverse_random_forest"]
    assert len(loo_cv_iteration_list) == experiment_data_df.shape[0], len(loo_cv_iteration_list)

    input_result_dict["evaluation"]["reverse_random_forest"] = evaluate_feature_selection(
        loo_cv_iteration_list,
        experiment_data_df,
        hold_out_test_data_df,
        seed_random_forest,
        "feature_subset_selection",
    )
    # check if all evaluation performance metrics for reverse rf are different
    for performance_metric in input_result_dict["evaluation"]["reverse_random_forest"]["performance_metrics_df"]:
        comparison_series = input_result_dict["evaluation"]["reverse_random_forest"]["performance_metrics_df"][
            performance_metric
        ]
        for key, value in input_result_dict["evaluation"].items():
            if "reverse" not in key:
                # compare two columns of a pandas dataframe
                assert not comparison_series.equals(value["performance_metrics_df"][performance_metric])


def evaluate_repeated_benchmark_experiments(
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
    # Initialize lists to store the random seeds to ensure the uniqueness of each random seed
    random_states_rf = []
    random_seeds_reverse_rf = []

    list_of_result_dicts = []

    # iterate over all repeated experiments
    for experiment_id, seed in zip(repeated_experiments, seeds, strict=True):
        # load the result dictionary for the given experiment
        file_path = os.path.join(result_dict_path, f"{experiment_id}_result_dict.pkl")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)

        # collect the random states for the standard random forest from the metadata
        random_states_rf.append(result_dict["standard_random_forest_meta_data"]["random_state"])
        if "reverse_random_forest_meta_data" in result_dict:
            # collect the random seeds for the reverse random forest from the metadata
            random_seeds_reverse_rf.append(result_dict["reverse_random_forest_meta_data"]["random_seeds"])

        # check if the result dictionary already contains the evaluation results
        if "evaluation" in result_dict:
            logging.info(f"Evaluation results for {experiment_id} already present. Deleting them.")
            # delete the evaluation results to reevaluate the experiment
            del result_dict["evaluation"]

        # evaluate the experiment
        evaluate_benchmark_experiment(result_dict, seed)
        list_of_result_dicts.append(result_dict)

    assert len(random_states_rf) == len(repeated_experiments), "Seeds are missing."
    assert len(set(random_states_rf)) == len(repeated_experiments), "Seeds have duplicates."
    check_random_seeds_equality(random_seeds_reverse_rf, repeated_experiments)
    return list_of_result_dicts


def evaluate_data_set(
    base_path: str,
    list_of_experiments: list[str],
    seeds: list[int],
    data_display_name: str,
    reload_evaluation_results: bool = False,
):
    """Evaluate the feature selection results of the experiment series of the given data set.

    Args:
        base_path: The base path to the pickled results files.
        list_of_experiments: A list of the experiment ids of repeated experiments to evaluate.
        seeds: A list of random seeds for the random forest classifier to ensure reproducibility.
        data_display_name: The name of the data set to display.
        reload_evaluation_results: A boolean indicating whether to reload the evaluation results. Defaults to False.

    Returns:
        The summarized evaluation results.
    """
    if reload_evaluation_results and Path(f"{base_path}/{data_display_name}_evaluation_results.pkl").exists():
        logging.info(f"Loading evaluation results for {data_display_name}")
        # load the pickled evaluation results
        with open(f"{base_path}/{data_display_name}_evaluation_results.pkl", "rb") as file:
            list_of_result_dicts = pickle.load(file)

    else:
        # evaluate the feature selection results for the given data
        logging.info(f"Evaluating {data_display_name}")
        list_of_result_dicts = evaluate_repeated_benchmark_experiments(base_path, list_of_experiments, seeds)

        # pickle the list of result dictionaries
        with open(f"{base_path}/{data_display_name}_evaluation_results.pkl", "wb") as file:
            pickle.dump(list_of_result_dicts, file)

    summarized_evaluation_results = summarize_evaluation_results(list_of_result_dicts)
    create_latex_table(summarized_evaluation_results, data_display_name)
    return summarized_evaluation_results


def check_random_seeds_equality(random_seeds_reverse_rf, repeated_experiments):
    """Check if the random seeds are equal for the repeated experiments.

    Args:
        random_seeds_reverse_rf: The random seeds for the reverse random forest.
        repeated_experiments: The list of repeated experiments.
    """
    if len(random_seeds_reverse_rf) != len(repeated_experiments):
        print("Random seeds are missing. Only found " + str(len(random_seeds_reverse_rf)))

    # flatten list of seeds to check if the random seeds are equal
    flattened_lists_of_seeds: list[int] = [seed for sublist in random_seeds_reverse_rf for seed in sublist]
    if len(set(flattened_lists_of_seeds)) != len(repeated_experiments) * 30:
        # find equal elements within the list
        equal_elements = [
            element for element in flattened_lists_of_seeds if flattened_lists_of_seeds.count(element) > 1
        ]
        print("total number of equal elements: ", len(equal_elements))
        print(set(equal_elements))
        print("number of equal elements: ", len(set(equal_elements)))


def summarize_evaluation_results(
    list_of_result_dicts: list[dict],
) -> pd.DataFrame:
    """Summarize the evaluation results for the given list of result dictionaries.

    Summarizes a list of result dictionaries containing evaluation results into a single pandas DataFrame.
    It aggregates performance metrics, stability, and the number of selected features for each feature selection
    method across multiple experiments. The function calculates the mean and standard deviation for these metrics
    and returns the summarized results in a DataFrame.

    Args:
        list_of_result_dicts: A list of result dictionaries containing the evaluation results.

    Returns:
        The summarized evaluation results.
    """
    summarized_result_dict: dict[str, dict] = {}
    intermediate_result_dict = summarized_result_dict.copy()

    for result_dict in list_of_result_dicts:
        for key, value in result_dict["evaluation"].items():
            assert isinstance(value["stability"], float)
            if key not in summarized_result_dict:
                summarized_result_dict[key] = {
                    "performance_metrics_df": value["performance_metrics_df"],
                    "stability": [value["stability"]],
                    "number of selected features": value["number of selected features"].tolist(),
                }
            else:
                summarized_result_dict[key]["performance_metrics_df"] = pd.concat(
                    [summarized_result_dict[key]["performance_metrics_df"], value["performance_metrics_df"]], axis=0
                )
                summarized_result_dict[key]["stability"].append(value["stability"])
                summarized_result_dict[key]["number of selected features"].extend(value["number of selected features"])

        mean_of_summarized_result_dict = {
            key: {
                "performance_metrics_df": value["performance_metrics_df"].mean(),
                "stability": np.mean(value["stability"]),
                "number of selected features": np.mean(value["number of selected features"]),
            }
            for key, value in summarized_result_dict.items()
        }
        # check if the mean of the lists in the intermediate_result_dict differ in every iteration
        if len(intermediate_result_dict) > 0:
            # check if the mean in mean_of_summarized_result_dict is different from the previous iteration
            for key, value in mean_of_summarized_result_dict.items():
                assert not intermediate_result_dict[key]["performance_metrics_df"].equals(
                    value["performance_metrics_df"]
                )
                assert intermediate_result_dict[key]["stability"] != value["stability"]
                assert (
                    intermediate_result_dict[key]["number of selected features"] != value["number of selected features"]
                )
        intermediate_result_dict = mean_of_summarized_result_dict.copy()

    # initialize the summarized dataframe with the columns of the performance metrics
    summarized_df = pd.DataFrame(
        columns=summarized_result_dict["reverse_random_forest"]["performance_metrics_df"].columns
    )
    for feature_selection_method_name, metrics in summarized_result_dict.items():
        if "shap" in feature_selection_method_name:
            continue

        summarized_df.loc[feature_selection_method_name] = metrics["performance_metrics_df"].mean()
        summarized_df.loc[f"{feature_selection_method_name}_std"] = metrics["performance_metrics_df"].std()
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
        # round the floats to three decimal places
        df_for_latex[column_name] = df_for_latex[column_name].apply(lambda x: f"{x:.3f}")

        for index_name in df_for_latex.index:
            if "std" not in index_name:
                # insert latex code and append std to mean values
                df_for_latex.loc[index_name, column_name] = (
                    f"${df_for_latex.loc[index_name, column_name]}"
                    f"\pm{df_for_latex.loc[index_name+'_std', column_name]}$"  # noqa: W605
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
            counter -= 3
        if counter > 0:
            print(
                df_for_latex[performance_metrics_list[-counter:]].to_latex(
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


def initialize_figure(data_names_list: list[str], performance_metric: str) -> tuple:
    """Initialize the figure for the given data names list and performance metric.

    Args:
        data_names_list: The list of names of the analysed data sets.
        performance_metric: The performance metric to display. Possible values are "Accuracy", "AUC",
            "Average Precision Score", "Cohen's Kappa", "Precision", "Sensitivity", "Specificity".

    Returns:
        The initialized figure, color map, and text positions dictionary.
    """
    # check if data_names_list contains repetitions of the same data set with a different correlation threshold
    if "thres" in data_names_list[0]:
        # intialize the figure with the given data name
        fig = go.Figure(
            layout_title_text="Different values for the correlation threshold parameter",
            # make background white
            layout={
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
            },
        )
        fig.update_xaxes(title="Stability of Feature Selection")
        fig.update_yaxes(title=performance_metric)
        fig.update_layout(margin={"l": 60, "r": 10, "t": 25, "b": 50})
        # # set color map
        # color_map = {
        #     "threshold 0.1": "white",
        #     "threshold 0.2": "lightgrey",
        #     "threshold 0.3": "grey",
        #     "threshold 0.4": "darkgrey",
        # }
        # # set the positions for the text labels
        # text_positions_dict = {
        #     "Leukemia threshold 0.1": "middle center",
        #     "Leukemia threshold 0.2": "middle center",
        #     "Leukemia threshold 0.3": "middle center",
        #     "Leukemia threshold 0.4": "middle center",
        #     "default": "middle center",
        # }
        color_map = text_positions_dict = None

    else:
        # set the colors for the different methods
        color_map = {
            "reverse random forest": "red",
            "without HPO": "black",
            "gini impurity": "green",
            "permutation": "blue",
        }
        # set the positions for the text labels
        text_positions_dict = {
            # text = gini, without HPO, permutation, reverse random forest
            "Colon Cancer": ["top center", "bottom right", "bottom right", "bottom right"],
            "Prostate Cancer": ["top center", "top left", "bottom center", "top right"],
            # text = permutation, gini, without HPO, reverse random forest
            "Leukemia": ["top left", "top center", "bottom left", "bottom center"],
            "default": ["bottom left", "bottom left", "bottom left", "bottom left"],
        }
        fig = make_subplots(
            rows=len(data_names_list),
            cols=1,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=data_names_list,
            x_title="Stability of Feature Selection",
            y_title=performance_metric,
            horizontal_spacing=0,
            vertical_spacing=0.03,
        )
        # Update the axes and layout
        fig.update_xaxes(range=[0, 1.001])
        fig.update_yaxes(range=[0, 1.2])
        fig.update_layout(showlegend=False, margin={"l": 60, "r": 10, "t": 25, "b": 50})

    return fig, color_map, text_positions_dict


def visualize_results(
    summarized_evaluation_results_df: pd.DataFrame,
    data_display_name: str,
    performance_metric: str,
    row_number: int,
    fig=None,
    color_map=None,
    text_positions_dict=None,
):
    """Visualize the results for the given data set.

    Args:
        summarized_evaluation_results_df: Dataframe containing the summarized evaluation results of repeated experiments
            for one data set.
        data_display_name: The name of the data set to display.
        performance_metric: The performance metric to display.
        row_number: The row number of the subplot.
        fig: The plotly figure to extend. Defaults to None.
        color_map: The color map for the different methods. Defaults to None.
        text_positions_dict: The text positions for the method labels. Defaults to None.

    Returns:
        The updated plotly figure.
    """
    # extract the standard deviation values for each performance metric
    column_lists = [
        [
            summarized_evaluation_results_df.loc[idx, column_name]
            for idx in summarized_evaluation_results_df.index
            if "std" in idx
        ]
        for column_name in summarized_evaluation_results_df.columns
    ]
    # delete rows with "_std" from the index
    summarized_result_df = summarized_evaluation_results_df.drop(
        index=[index for index in summarized_evaluation_results_df.index if "std" in index]
    )
    # insert std as columns with column_name_std to the dataframe
    for std_list, column in zip(column_lists, summarized_result_df.columns, strict=True):
        summarized_result_df.loc[:, f"{column}_std"] = std_list

    summarized_result_df["method"] = summarized_result_df.index

    # check if the data set is a correlation threshold parameter experiment
    if "threshold" in data_display_name:
        # drop all methods except for the reverse random forest
        summarized_result_df = summarized_result_df.loc[["reverse_random_forest"]]

        # rename the method name and append the current threshold
        summarized_result_df.loc["reverse_random_forest", "method"] = f"{' '.join(data_display_name.split(' ')[1:])}"

        # round number of selected features to two decimal places
        summarized_result_df["number of selected features"] = summarized_result_df["number of selected features"].round(
            2
        )

        fig.add_trace(
            go.Scatter(
                x=summarized_result_df["stability"],
                y=summarized_result_df[performance_metric],
                error_y={
                    "type": "data",
                    "array": summarized_result_df[f"{performance_metric}_std"],
                    "color": "lightgrey",
                },
                error_x={
                    "type": "data",
                    "array": summarized_result_df["stability_std"],
                    "color": "lightgrey",
                },
                mode="markers+text",
                text=summarized_result_df["number of selected features"],
                # set size corresponding to the number of selected features
                marker_size=summarized_result_df["number of selected features"],
                # add text as element to legend
                name=summarized_result_df["method"].to_numpy[0],
                # marker={"color": summarized_result_df["method"].apply(lambda x: color_map[x])},
            ),
        )
    else:
        text_positions = text_positions_dict.get(data_display_name, text_positions_dict["default"])
        assert len(text_positions) == len(summarized_result_df["method"])
        assert isinstance(text_positions, list)
        # rename the method names for better readability
        for method in summarized_result_df["method"]:
            if method == "reverse_random_forest":
                summarized_result_df.loc[method, "method"] = "reverse random forest"
            elif method == "gini_impurity_default_parameters":
                summarized_result_df.loc[method, "method"] = "without HPO"
            elif method == "gini_impurity":
                summarized_result_df.loc[method, "method"] = "gini impurity"

        fig.add_trace(
            go.Scatter(
                x=summarized_result_df["stability"],
                y=summarized_result_df[performance_metric],
                error_y={
                    "type": "data",
                    "array": summarized_result_df[f"{performance_metric}_std"],
                    "color": "lightgrey",
                },
                error_x={
                    "type": "data",
                    "array": summarized_result_df["stability_std"],
                    "color": "lightgrey",
                },
                mode="markers+text",
                text=summarized_result_df["method"],
                textposition=text_positions,
                marker={"color": summarized_result_df["method"].apply(lambda x: color_map[x])},
            ),
            row=row_number,
            col=1,
        )
    return fig


def main():
    """Run the evaluation for the given experiments."""
    # seeds for the random forest classifier for each repeated experiment
    seeds = [2, 424, 20000]
    # seeds = [2]

    # base path to the pickled results files
    base_path = "../../results"

    # evaluate the feature selection results for the given data
    experiments_dict = {
        "Colon Cancer": [
            "colon_0_shuffle_seed_None_rf",
            "colon_1_shuffle_seed_None_rf",
            "colon_2_shuffle_seed_None_rf",
        ],
        # "Prostate Cancer": [
        #     "prostate_0_shuffle_seed_None_thres_01_ranger",
        #     "prostate_1_shuffle_seed_None_thres_01_ranger",
        #     "prostate_2_shuffle_seed_None_thres_01_ranger",
        # ],
        # "Leukemia": [
        #     "leukemia_big_0_shuffle_seed_None_thres_04_ranger",
        # ],
        "Prostate Cancer": [
            "prostate_0_shuffle_seed_None_rf",
            "prostate_1_shuffle_seed_None_rf",
            "prostate_2_shuffle_seed_None_rf",
        ],
        "Leukemia": [
            "leukemia_big_0_shuffle_seed_None_rf",
            "leukemia_big_1_shuffle_seed_None_rf",
            "leukemia_big_2_shuffle_seed_None_rf",
        ],
        # "Leukemia threshold 0.1": [
        #     "leukemia_big_0_shuffle_seed_None_thres_01_ranger",
        #     # "leukemia_big_1_shuffle_seed_None_thres_01_ranger",
        #     # "leukemia_big_2_shuffle_seed_None_thres_01_ranger",
        # ],
        # "Leukemia threshold 0.2": [
        #     "leukemia_big_0_shuffle_seed_None_rf",
        #     # "leukemia_big_1_shuffle_seed_None_rf",
        #     # "leukemia_big_2_shuffle_seed_None_rf",
        # ],
        # "Leukemia threshold 0.3": [
        #     "leukemia_big_0_shuffle_seed_None_thres_03_ranger",
        #     # "leukemia_big_1_shuffle_seed_None_thres_01_ranger",
        #     # "leukemia_big_2_shuffle_seed_None_thres_01_ranger",
        # ],
        # "Leukemia threshold 0.4": [
        #     "leukemia_big_0_shuffle_seed_None_thres_04_ranger",
        #     # "leukemia_big_1_shuffle_seed_None_thres_01_ranger",
        #     # "leukemia_big_2_shuffle_seed_None_thres_01_ranger",
        # ],
        # "Random Noise Normal": [
        #     "random_noise_normal_0_shuffle_seed_None_ranger",
        #     "random_noise_normal_1_shuffle_seed_None_ranger",
        #     "random_noise_normal_2_shuffle_seed_None_ranger",
        # ],
        # "Random Noise Lognormal": [
        #     "random_noise_lognormal_0_shuffle_seed_None_ranger",
        #     "random_noise_lognormal_1_shuffle_seed_None_ranger",
        #     "random_noise_lognormal_2_shuffle_seed_None_ranger",
        # ],
    }
    # extract all experiments without random noise
    data_names_list = [data_display_name for data_display_name in experiments_dict if "andom" not in data_display_name]

    # define the performance metric to plot
    performance_metric = "Average Precision Score"

    # initialize the figure for the feature selection method comparison
    fig, color_map, text_positions_dict = initialize_figure(data_names_list, performance_metric)

    row_number_in_subplot = 1
    for data_display_name, experiment_id_list in experiments_dict.items():
        summarized_evaluation_results_df = evaluate_data_set(
            base_path, experiment_id_list, seeds, data_display_name=data_display_name, reload_evaluation_results=True
        )
        if data_display_name in data_names_list:
            # visualize the results
            fig = visualize_results(
                summarized_evaluation_results_df,
                data_display_name,
                performance_metric,
                row_number_in_subplot,
                fig,
                color_map,
                text_positions_dict,
            )
            row_number_in_subplot += 1

    fig.show()

    # # save figure
    # # pio.write_image(fig, f"{base_path}/images/{performance_metric}.pdf", width=3508, height=2480)
    pio.write_image(fig, f"{base_path}/images/{performance_metric}.pdf", width=1000, height=600)
    # pio.write_image(fig, f"{base_path}/images/{performance_metric}.pdf", width=800, height=600)


if __name__ == "__main__":
    main()
