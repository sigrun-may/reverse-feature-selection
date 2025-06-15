# Copyright (c) 2025 Sigrun May,
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    balance_sample_size_of_hold_out_data_single_df,
    load_train_holdout_data_for_balanced_train_sample_size,
)
from feature_selection_benchmark.feature_selection_evaluation.stability_estimator import calculate_stability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_JOBS = 1


def train_and_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_importance,
    seed: int,
) -> tuple:
    """Trains a model and predicts the labels for the tests data.

    Args:
        train_data: The training data.
        test_data: The tests data.
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
    """Calculate the performance metrics for the given hold-out tests data.

    Args:
        train_data_df: The training data.
        feature_subset_selection: The selected feature subset. Must be array_like for numpy.
        hold_out_test_data_df: The hold-out tests data.
        shuffle_seed_hold_out_data: The random seed for shuffling the hold-out tests data.
        seed_for_random_forest: The random seed for reproducibility of the random forest.

    Returns:
        The performance metrics for the given hold-out tests data.
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

    balanced_hold_out_test_data_df = balance_sample_size_of_hold_out_data_single_df(
        hold_out_test_data_df, shuffle_seed=shuffle_seed_hold_out_data, number_of_balanced_samples=14
    )
    y_predict, y_predict_proba, y_true = train_and_predict(
        train_data_df,
        balanced_hold_out_test_data_df,
        feature_subset_selection,
        seed_for_random_forest,
    )
    # extract the probability for the positive class for briar score calculation
    probability_positive_class = np.array([proba[1] for proba in y_predict_proba])
    performance_metrics_dict = {
        # for the given hold out tests set the healthy control 0 is the minority class
        "Average Precision Score": average_precision_score(y_true, y_predict),
        "AUC": roc_auc_score(y_true, probability_positive_class),
        "Accuracy": accuracy_score(y_true, y_predict),
        "Sensitivity": recall_score(y_true, y_predict),
        "Specificity": recall_score(y_true, y_predict, pos_label=hold_out_test_data_df["label"].unique()[0]),
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
    """Calculate the performance metrics for the given hold-out tests data.

    Args:
        train_data_df: The training data.
        feature_subset_selection: The selected feature subset. Must be array_like for numpy.
        hold_out_test_data_df: The hold-out tests data.
        seed_for_random_forest: The random seed for reproducibility of the random forest.

    Returns:
        The performance metrics for the given hold-out tests data.
    """
    # parallelize iterations of shuffling the hold out tests data with joblib
    return joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(calculate_performance_metrics)(
            train_data_df=train_data_df,
            feature_subset_selection=feature_subset_selection,
            hold_out_test_data_df=hold_out_test_data_df,
            shuffle_seed_hold_out_data=(hold_out_shuffle_iteration + 10000),
            seed_for_random_forest=seed_for_random_forest,
        )
        # TODO for hold_out_shuffle_iteration in range(100)
        for hold_out_shuffle_iteration in range(2)
    )


def evaluate_performance(
    feature_importance_matrix: np.ndarray,
    experiment_input_data_df: pd.DataFrame,
    hold_out_test_data_df: pd.DataFrame,
    seed_for_random_forest: int,
) -> pd.DataFrame:
    """Evaluate the performance of the feature selection methods based on the feature importance matrix.

    Args:
        feature_importance_matrix: A numpy array of shape (number of iterations, number of features) containing the
            feature importance scores for each cross-validation iteration.
        experiment_input_data_df: The input data for the experiment.
        hold_out_test_data_df: The hold-out test data for evaluating the performance of the feature selection methods.
        seed_for_random_forest: The random seed for reproducibility of the random forest.
    Returns:
        A dictionary containing the evaluation results.
    """
    performance_metrics_list = []
    for loo_idx in range(feature_importance_matrix.shape[0]):
        train_data_df = experiment_input_data_df.drop(index=loo_idx)
        assert train_data_df.shape[0] == experiment_input_data_df.shape[0] - 1

        performance_metrics_list.extend(
            calculate_performance_metrics_on_shuffled_hold_out_subset(
                train_data_df,
                feature_importance_matrix[loo_idx],
                hold_out_test_data_df,
                seed_for_random_forest,
            )
        )
    performance_metrics_df = pd.DataFrame(performance_metrics_list)
    return performance_metrics_df


def get_importance_matrix(loo_cv_iteration_list, feature_selection_method_key) -> np.ndarray:
    """Extract the feature importance matrix from the cross-validation iterations.

    The function constructs a matrix where each row corresponds to a cross-validation iteration and each column
    corresponds to a feature's importance score. This is useful for analyzing the stability and performance of
    feature selection methods across multiple cross-validation iterations. It allows for the assessment of how
    consistently features are selected across different subsets of the data, which is crucial for understanding the
    reliability of feature selection methods in high-dimensional datasets.
    Args:
        loo_cv_iteration_list: List of cross-validation iterations, each containing feature importance data. Assumes
            that each entry in loo_cv_iteration_list is either a dictionary or a pandas dataframe containing the feature
            importance data under the specified feature_selection_method_key. If the feature_selection_method_key is
            "permutation", it sets all negative values to 0, as permutation importance can yield negative values.
            For other methods, it asserts that all values are non-negative.
        feature_selection_method_key: Key to access the feature importance data in each iteration.
    Returns:
        A numpy array of shape (number of iterations, number of features) containing the feature importance scores.

    Raises:
        ValueError: If the feature_selection_method_key is not found in the cross-validation iterations.
        AssertionError: If the minimum value in the final feature importance matrix is negative.
    Returns:
        A numpy array of shape (number of iterations, number of features) containing the feature importance scores.
    """
    # initialize a feature importance matrix with the shape of (number of iterations, number of features)
    feature_importance_matrix = np.empty(
        (len(loo_cv_iteration_list), len(loo_cv_iteration_list[0][feature_selection_method_key]))
    )
    for loo_idx, cv_iteration_result in enumerate(loo_cv_iteration_list):
        feature_importances = cv_iteration_result[feature_selection_method_key]
        feature_importance_matrix[loo_idx] = feature_importances

    if "permutation" in feature_selection_method_key:
        # set all negative values to 0
        feature_importance_matrix[feature_importance_matrix < 0] = 0

    assert np.min(feature_importance_matrix) >= 0
    return feature_importance_matrix


def extract_feature_subset_selection_matrices_and_wall_times(result_dict: dict):
    """Extract the feature subset selection and the wall times from the cross-validation iterations.

    Args:
        result_dict: Dictionary containing the results of the cross-validation iterations for different methods.
    """
    for method_name, cv_iteration_list in result_dict.items():
        if "meta_data" in method_name:
            continue

        # get the feature importance matrix
        if "reverse" in method_name:
            result_dict["evaluation"]["reverse"] = {
                "importance_matrix": get_importance_matrix(
                    loo_cv_iteration_list=cv_iteration_list, feature_selection_method_key="feature_subset_selection"
                ),
                # skip the first value
                "wall_times": result_dict[f"{method_name}_meta_data"]["benchmark"]["wall_times"][1:],
            }
        elif "standard" in method_name:
            result_dict["evaluation"]["gini"] = {
                "importance_matrix": get_importance_matrix(
                    loo_cv_iteration_list=cv_iteration_list, feature_selection_method_key="gini_impurity"
                ),
                # skip the first value
                "wall_times": result_dict[f"{method_name}_meta_data"]["benchmark"]["wall_times"][1:],
            }
            result_dict["evaluation"]["gini_default"] = {
                "importance_matrix": get_importance_matrix(
                    loo_cv_iteration_list=cv_iteration_list,
                    feature_selection_method_key="gini_impurity_default_parameters",
                )
            }
        elif "ranger" in method_name:
            result_dict["evaluation"]["permutation"] = {
                "importance_matrix": get_importance_matrix(
                    loo_cv_iteration_list=cv_iteration_list, feature_selection_method_key="permutation"
                ),
                # skip the first value
                "wall_times": result_dict[f"{method_name}_meta_data"]["benchmark"]["wall_times"][1:],
            }


def evaluate_experiment(result_dict: dict, experiment_id: str) -> dict:
    """Evaluate the experiment and extract the feature subset selection matrices.

    Args:
        result_dict: Dictionary containing the results of the cross-validation iterations for different methods.
        experiment_id: The ID of the experiment to evaluate.
    Returns:
        A dictionary containing the evaluation results, including the feature importance matrices, stability, and
        performance metrics.
    """
    # load the original experiment input data and the hold-out tests data
    experiment_input_data_df, hold_out_test_data_df = load_train_holdout_data_for_balanced_train_sample_size(
        result_dict["reverse_random_forest_meta_data"]
    )
    result_dict["evaluation"] = {}
    extract_feature_subset_selection_matrices_and_wall_times(result_dict)
    assert result_dict["evaluation"] != {}, "No evaluation data found in the result dictionary."

    for method in result_dict["evaluation"]:
        if "importance_matrix" not in result_dict["evaluation"][method]:
            raise ValueError(f"Result dictionary does not contain 'importance_matrix' key for method {method}.")
        if result_dict["evaluation"][method]["importance_matrix"] is None:
            raise ValueError(f"Importance matrix for method {method} is None.")
        if len(result_dict["evaluation"][method]["importance_matrix"]) == 0:
            raise ValueError(f"Importance matrix for method {method} is empty.")

        # if no features were selected, stability cannot be calculated
        if "noise" not in experiment_id.lower():
            # calculate the stability of the feature importance matrix
            result_dict["evaluation"][method]["stability"] = calculate_stability(
                result_dict["evaluation"][method]["importance_matrix"]
            )
        # calculate number of selected features per cv iteration
        result_dict["evaluation"][method]["number_of_selected_features"] = np.count_nonzero(
            result_dict["evaluation"][method]["importance_matrix"], axis=1
        ).tolist()

        # calculate the performance of the feature selection method
        performance_metrics_df = evaluate_performance(
            feature_importance_matrix=result_dict["evaluation"][method]["importance_matrix"],
            experiment_input_data_df=experiment_input_data_df,
            # TODO  hold_out_test_data_df=hold_out_test_data_df,
            hold_out_test_data_df=experiment_input_data_df,
            seed_for_random_forest=result_dict["reverse_random_forest_meta_data"]["random_seeds"][0],
        )
        # store the performance metrics in the result dictionary
        result_dict["evaluation"][method]["performance_metrics"] = performance_metrics_df
    return result_dict


def evaluate_repeated_experiments(experiment_id_list, result_dict_directory_path) -> tuple[list[dict], dict]:
    """Evaluate the repeated experiments and extract the feature subset selection matrices.

    Args:
        experiment_id_list: List of experiment IDs to evaluate.
        result_dict_directory_path: Path to the directory containing the result dictionaries for the experiments.
    Returns:
        A tuple containing:
            - A list of dictionaries containing the evaluated results for each experiment.
            - A dictionary summarizing the results across all experiments.
    """
    evaluated_result_dict_list = []
    summarized_result_dict: dict[str, dict] = {}
    # iterate over the repeated experiments
    for experiment_id in experiment_id_list:
        # load the result dictionary for the given experiment
        file_path = os.path.join(result_dict_directory_path, f"{experiment_id}_result_dict.pkl")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)

        # evaluate the experiment and extract the feature subset selection matrices
        evaluated_result_dict = evaluate_experiment(result_dict=result_dict, experiment_id=experiment_id)
        evaluated_result_dict_list.append(evaluated_result_dict)

        for method, evaluation_results_dict in evaluated_result_dict["evaluation"].items():
            if method not in summarized_result_dict:
                summarized_result_dict[method] = {}
            for key, value in evaluation_results_dict.items():
                if key == "importance_matrix":
                    continue
                if key not in summarized_result_dict[method]:
                    # initialize the key in the summarized result dictionary
                    summarized_result_dict[method][key] = value
                elif isinstance(value, pd.DataFrame):
                    # concatenate performance metrics DataFrame
                    summarized_result_dict[method][key] = pd.concat(
                        [summarized_result_dict[method][key], value], ignore_index=True
                    )
                else:
                    summarized_result_dict[method][key].extend(value)

    return evaluated_result_dict_list, summarized_result_dict


def print_results(evaluation_dict):
    """Print the benchmark results in a readable format.

    Args:
        evaluation_dict: Dictionary containing the evaluation results for different datasets and feature selection
        methods.
    """
    print("Benchmark results:")
    for data_name, feature_selection_methods_benchmark_dict in evaluation_dict.items():
        print(f"{data_name}: ##############################################################################")
        for feature_selection_method, benchmark_dict in feature_selection_methods_benchmark_dict.items():
            if feature_selection_method == "number_of_input_features":
                continue
            for metric, values in benchmark_dict.items():
                if metric == "importance_matrix":
                    continue
                if isinstance(values, pd.DataFrame):
                    print(f"  {feature_selection_method} {metric}:")
                    print(values.describe().T[["mean", "std"]])
                    print(" ")
                elif len(values) > 0:
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    print(f"  {feature_selection_method} {metric}: {mean_value} ± {std_value}")
                else:
                    raise ValueError(f"Metric {metric} is empty.")


def visualize_runtime_benchmark(benchmark_dict):
    """Visualize the runtime benchmark results.

    Args:
        benchmark_dict: Dictionary containing the benchmark results for different datasets and feature selection
        methods.
    """
    summarized_benchmark_dict = {}
    methods = []
    for data_name, methods_benchmark_dict in benchmark_dict.items():
        summarized_benchmark_dict[data_name] = {
            "number_of_input_features": benchmark_dict[data_name]["number_of_input_features"]
        }
        for feature_selection_method, method_benchmark_dict in methods_benchmark_dict.items():
            if isinstance(method_benchmark_dict, dict):
                methods.append(feature_selection_method)
                for metric, values in method_benchmark_dict.items():
                    if isinstance(values, pd.DataFrame):
                        for column in values.columns:
                            summarized_benchmark_dict[data_name][f"{feature_selection_method}_{column}_mean"] = np.mean(
                                values[column]
                            )
                            summarized_benchmark_dict[data_name][f"{feature_selection_method}_{column}_std"] = np.std(
                                values[column]
                            )
                    elif isinstance(values, list):
                        summarized_benchmark_dict[data_name][f"{feature_selection_method}_{metric}_mean"] = np.mean(
                            values
                        )
                        summarized_benchmark_dict[data_name][f"{feature_selection_method}_{metric}_std"] = np.std(
                            values
                        )

    # Convert the benchmark_dict to a DataFrame
    summarized_benchmark_df = pd.DataFrame(summarized_benchmark_dict).T

    # Plot the summarized benchmark results
    for method in set(methods):
        if "default" in method:
            continue
        yerr = summarized_benchmark_df[f"{method}_wall_times_std"]
        xerr = summarized_benchmark_df[f"{method}_number_of_selected_features_std"]
        plt.errorbar(
            x=summarized_benchmark_df[f"{method}_number_of_selected_features_mean"],
            y=summarized_benchmark_df[f"{method}_wall_times_mean"],
            yerr=yerr,
            xerr=xerr,
            label=method.replace("_", " ").title(),
            marker="o",
            capsize=5,
        )
    plt.title("Benchmark: Wall Time Mean vs Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Wall Time Mean (s)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.show()


def evaluate_benchmark_experiment_grid():
    """Evaluate the benchmark experiment grid."""
    experiments_dict = {
        "Random Noise Lognormal": [
            "tiny_test_dataset_random_noise_lognormal_0_shuffle_seed_13_stdrf",
            "tiny_test_dataset_random_noise_lognormal_1_shuffle_seed_13_stdrf",
        ],
        "Random Noise Normal": [
            "tiny_test_dataset_random_noise_normal_0_shuffle_seed_13_stdrf",
            #     "random_noise_normal_0_shuffle_seed_13_stdrf",
            #     "random_noise_normal_1_shuffle_seed_13_stdrf",
            #     "random_noise_normal_2_shuffle_seed_13_stdrf",
        ],
        # "Random Noise Lognormal": [
        #     "random_noise_lognormal_0_shuffle_seed_13_stdrf",
        #     "random_noise_lognormal_1_shuffle_seed_13_stdrf",
        #     "random_noise_lognormal_2_shuffle_seed_13_stdrf",
        # ],
        # "Colon Cancer": [
        #     "colon_0_shuffle_seed_13_stdrf",
        #     "colon_1_shuffle_seed_13_stdrf",
        #     "colon_2_shuffle_seed_13_stdrf",
        # ],
        # "Leukemia Cancer": [
        #     "leukemia_big_0_shuffle_seed_13_stdrf",
        #     "leukemia_big_1_shuffle_seed_13_stdrf",
        #     "leukemia_big_2_shuffle_seed_13_stdrf",
        # ],
        # "Random Noise Normal": [
        #     "random_noise_normal_0_shuffle_seed_None",
        #     "random_noise_normal_1_shuffle_seed_None",
        #     "random_noise_normal_2_shuffle_seed_None",
        # ],
        # "Random Noise Lognormal": [
        #     "random_noise_lognormal_0_shuffle_seed_None_ranger",
        #     "random_noise_lognormal_1_shuffle_seed_None_ranger",
        #     "random_noise_lognormal_2_shuffle_seed_None_ranger",
        # ],
        # "Colon Cancer": [
        #     "colon_0_shuffle_seed_None_stdrf",
        #     "colon_1_shuffle_seed_None_stdrf",
        #     "colon_2_shuffle_seed_None_stdrf",
        # ],
        # "Prostate Cancer": [
        #     "prostate_0_shuffle_seed_None_stdrf",
        #     "prostate_1_shuffle_seed_None_stdrf",
        #     "prostate_2_shuffle_seed_None_stdrf",
        # ],
        # "Leukemia Cancer": [
        #     "leukemia_big_0_shuffle_seed_None_stdrf_2025_05_20",
        #     # "leukemia_big_1_shuffle_seed_None_stdrf",
        #     # "leukemia_big_2_shuffle_seed_None_stdrf",
        # ],
        # "Prostate Cancer": [
        #     "prostate_0_shuffle_seed_None_stdrf",
        #     "prostate_1_shuffle_seed_None_stdrf",
        #     "prostate_2_shuffle_seed_None_stdrf",
        # ],
    }
    # result_dict_directory_path = "../../results"
    result_dict_directory_path = "results/tiny_test"
    # result_dict_directory_path = "../../results/benchmark_2025_06_03"
    # result_dict_directory_path = "../../results/colon_2025_05_15"
    # result_dict_directory_path = "../../results/prostate_2025-05-26"

    evaluation_dict = {}
    summaries_dict = {}  # number of input features for the benchmark
    # iterate over the analysed datasets
    for data_display_name, experiment_id_list in experiments_dict.items():
        list_of_evaluated_results, summarized_result_dict = evaluate_repeated_experiments(
            experiment_id_list, result_dict_directory_path
        )
        evaluation_dict[data_display_name] = list_of_evaluated_results
        summaries_dict[data_display_name] = summarized_result_dict
        summaries_dict[data_display_name]["number_of_input_features"] = list_of_evaluated_results[0][
            "reverse_random_forest"
        ][0].shape[1]

    # print_results(summaries_dict)

    # save the summary dictionary to a pickle file
    summary_file_path = Path(f"{result_dict_directory_path}/benchmark_summary.pkl")
    with open(summary_file_path, "wb") as summary_file:
        pickle.dump(summaries_dict, summary_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Summary saved to {summary_file_path}")
    # visualize_runtime_benchmark(benchmark_dict)


# # unpickle summary file
# summary_file_path = Path("results/tiny_test/benchmark_summary.pkl")
# with open(summary_file_path, "rb") as summary_file:
#     summary_dict = pickle.load(summary_file)
# print_results(summary_dict)
# visualize_runtime_benchmark(summary_dict)

evaluate_benchmark_experiment_grid()
