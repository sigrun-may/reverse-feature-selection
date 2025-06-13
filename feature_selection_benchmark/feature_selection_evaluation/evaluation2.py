import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_selection_benchmark.feature_selection_evaluation import stability_estimator
from feature_selection_benchmark.feature_selection_evaluation.stability_estimator import calculate_stability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_importance_matrix(loo_cv_iteration_list, feature_selection_method_key) -> np.ndarray:
    """Extract the feature importance matrix from the cross-validation iterations.

    The function constructs a matrix where each row corresponds to a cross-validation iteration and each column
    corresponds to a feature's importance score. This is useful for analyzing the stability and performance of
    feature selection methods across multiple cross-validation iterations. It allows for the assessment of how consistently
    features are selected across different subsets of the data, which is crucial for understanding the reliability of
    feature selection methods in high-dimensional datasets.
    Args:
        loo_cv_iteration_list: List of cross-validation iterations, each containing feature importance data. Assumes that
            each entry in loo_cv_iteration_list is either a dictionary or a pandas dataframe containing the feature
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
    Example:
    >>> loo_cv_iteration_list = [
    ...     {"feature_subset_selection": [0.1, 0.2, 0.3], "gini_impurity": [0.4, 0.5, 0.6]},
    ...     {"feature_subset_selection": [0.2, 0.1, 0.4], "gini_impurity": [0.5, 0.6, 0.7]},
    ... ]
    >>> feature_selection_method_key = "feature_subset_selection"
    >>> importance_matrix = get_importance_matrix(loo_cv_iteration_list, feature_selection_method_key)
    >>> print(importance_matrix)
    >>> # Output:
    >>> # [[0.1 0.2 0.3]
    >>> #  [0.2 0.1 0.4]]
    """
    # initialize a feature importance matrix with the shape of (number of iterations, number of features)
    feature_importance_matrix = np.empty_like(
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


def extract_feature_subset_selection_matrices(result_dict: dict):
    """Extract the feature subset selection from the cross-validation iterations.

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
                )
            }
        elif "standard" in method_name:
            result_dict["evaluation"]["gini"] = {
                "importance_matrix": get_importance_matrix(
                    loo_cv_iteration_list=cv_iteration_list, feature_selection_method_key="gini_impurity"
                )
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
                )
            }


def calculate_feature_selection_stability(result_dict: dict):
    """Calculate the feature selection stability for each method within the given result dictionary.

    Args:
        result_dict: Dictionary containing the importance matrices for different methods.
    """
    for method in result_dict["evaluation"].keys():
        if "importance_matrix" not in result_dict["evaluation"][method]:
            raise ValueError(f"Result dictionary does not contain 'importance_matrix' key for method {method}.")
        importance_matrix = result_dict["evaluation"][method]["importance_matrix"]
        if importance_matrix is None or len(importance_matrix) == 0:
            raise ValueError(f"Importance matrix for method {method} is empty or None.")
        # calculate the stability of the feature importance matrix
        result_dict["evaluation"][method]["stability"] = calculate_stability(importance_matrix)


def calculate_number_of_selected_features(result_dict: dict):
    """Calculate the number of selected features for each method within the given result dictionary.

    Args:
        result_dict: Dictionary containing the importance matrices for different methods.
    """
    for method in result_dict["evaluation"].keys():
        if "importance_matrix" not in result_dict["evaluation"][method]:
            raise ValueError(f"Result dictionary does not contain 'importance_matrix' key for method {method}.")
        importance_matrix = result_dict["evaluation"][method]["importance_matrix"]
        if importance_matrix is None or len(importance_matrix) == 0:
            raise ValueError(f"Importance matrix for method {method} is empty or None.")
        # calculate the number of selected features
        result_dict["evaluation"][method]["number_of_selected_features"] = np.count_nonzero(importance_matrix, axis=1)


def evaluate_performance(feature_importance_matrix: np.ndarray) -> dict:
    pass
    #TODO: Implement the performance evaluation logic based on the feature importance matrix.

def evaluate_experiment(result_dict: dict, experiment_id: str):

    result_dict["evaluation"] = {}
    extract_feature_subset_selection_matrices(result_dict)

    # if no features were selected, stability cannot be calculated
    if not "noise" in experiment_id.lower():
        # calculate the stability of the feature importance matrix
        result_dict["evaluation"]["stability"] = stability_estimator.calculate_stability(
            result_dict["evaluation"]["reverse"]["importance_matrix"]
        )
    # calculate number of selected features per cv iteration
    calculate_number_of_selected_features(result_dict)

    # calculate the performance of the feature selection methods
    return result_dict


def visualize_runtime_benchmark(benchmark_dict):
    summarized_benchmark_dict = {}
    methods = []
    for data_name, methods_benchmark in benchmark_dict.items():
        summarized_benchmark_dict[data_name] = {"number_of_features": methods_benchmark["number_of_features"]}
        for method_benchmark, values_list in methods_benchmark.items():
            if method_benchmark == "number_of_features":
                continue

            # strip the "_wall_time" or "_feature_count" suffix
            methods.append(method_benchmark.replace("_wall_time", "").replace("_feature_count", ""))

            summarized_benchmark_dict[data_name][f"{method_benchmark}_mean"] = np.mean(values_list)
            summarized_benchmark_dict[data_name][f"{method_benchmark}_std"] = np.std(values_list)

    # Convert the benchmark_dict to a DataFrame
    summarized_benchmark_df = pd.DataFrame(summarized_benchmark_dict).T
    # print(summarized_benchmark_df[["number_of_features", "reverse_random_forest_feature_count_mean", "reverse_random_forest_wall_time_mean"]])
    print(summarized_benchmark_df[["reverse_random_forest_feature_count_mean", "reverse_random_forest_wall_time_mean"]])
    # Plot the summarized benchmark results
    for method in set(methods):
        yerr = summarized_benchmark_df[f"{method}_wall_time_std"]
        plt.errorbar(
            x=summarized_benchmark_df["number_of_features"],
            y=summarized_benchmark_df[f"{method}_wall_time_mean"],
            yerr=yerr,
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


def get_benchmark_for_experiment(result_dict: dict, benchmark_dict: dict) -> dict:
    """Extract the benchmark results from the result dictionary.

    Args:
        result_dict: Dictionary containing the results of the cross-validation iterations for different methods.
    Returns:
        A dictionary containing the benchmark results.
    """
    if "evaluation" not in result_dict:
        raise ValueError(f"Result dictionary does not contain 'evaluation' key.")

    # iterate over the result dictionary and extract the benchmark results
    for key, meta_data_dict in result_dict.items():
        if "meta_data" not in key:
            continue
        if "benchmark" not in meta_data_dict:
            logger.warning(f"No benchmark data found for {key}.")
            continue

        # iterate over the benchmarks in the result dictionary and store the wall times and feature counts
        for selection_function_name, method_benchmark in meta_data_dict["benchmark"].items():
            if method_benchmark["wall_times"] is None:
                logger.warning(f"Wall times for {selection_function_name} are None in {key}.")
                continue
            if f"{selection_function_name}_wall_times" not in benchmark_dict:
                benchmark_dict[f"{selection_function_name}_wall_times"] = []
            benchmark_dict[f"{selection_function_name}_wall_times"].append(
                method_benchmark["wall_times"])

            # TODO remove duration for default random forest

            # add feature count to the benchmark dictionary
            if f"{selection_function_name}_feature_count" not in benchmark_dict:
                benchmark_dict[f"{selection_function_name}_feature_count"] = []

            if "reverse" in selection_function_name:
                benchmark_dict[f"{selection_function_name}_feature_count"].append(
                    result_dict["evaluation"]["reverse"]["number_of_selected_features"]
                )
            elif "standard" in selection_function_name:
                benchmark_dict[f"{selection_function_name}_feature_count"].append(
                    result_dict["evaluation"]["gini"]["number_of_selected_features"]
                )
            elif "ranger" in selection_function_name:
                benchmark_dict[f"{selection_function_name}_feature_count"].append(
                    result_dict["evaluation"]["ranger"]["number_of_selected_features"]
                )
            else:
                logger.warning(f"Unknown selection function name: {selection_function_name}.")
                continue
    return benchmark_dict


def evaluate_repeated_experiments(experiment_id_list, result_dict_directory_path)-> tuple[list[dict], dict]:
    evaluated_result_dict_list = []
    benchmark_dict = {}
    # iterate over the repeated experiments
    for experiment_id in experiment_id_list:
        # load the result dictionary for the given experiment
        file_path = os.path.join(result_dict_directory_path, f"{experiment_id}_result_dict.pkl")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found.")

        with open(file_path, "rb") as file:
            result_dict = pickle.load(file)

        # evaluate the experiment and extract the feature subset selection matrices
        evaluated_result_dict_list.append(evaluate_experiment(result_dict, experiment_id))

        # initialize the benchmark dictionary for the experiment
        benchmark_dict["number_of_features"] = result_dict["reverse_random_forest"][0].shape[0]

        # extract the benchmark results from the result dictionary
        get_benchmark_for_experiment(result_dict, benchmark_dict)

    return evaluated_result_dict_list, benchmark_dict


def evaluate_benchmark_experiment_grid():

    experiments_dict = {
        "Random Noise Normal": [
            "random_noise_normal_0_shuffle_seed_13_stdrf",
            "random_noise_normal_1_shuffle_seed_13_stdrf",
            "random_noise_normal_2_shuffle_seed_13_stdrf",
        ],
        "Random Noise Lognormal": [
            "random_noise_lognormal_0_shuffle_seed_13_stdrf",
            "random_noise_lognormal_1_shuffle_seed_13_stdrf",
            "random_noise_lognormal_2_shuffle_seed_13_stdrf",
        ],
        "Colon Cancer": [
            "colon_0_shuffle_seed_13_stdrf",
            "colon_1_shuffle_seed_13_stdrf",
            "colon_2_shuffle_seed_13_stdrf",
        ],
        "Leukemia Cancer": [
            "leukemia_big_0_shuffle_seed_13_stdrf",
            "leukemia_big_1_shuffle_seed_13_stdrf",
            "leukemia_big_2_shuffle_seed_13_stdrf",
        ],
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
    result_dict_directory_path = "../../results/benchmark_2025_06_03"
    # result_dict_directory_path = "../../results/colon_2025_05_15"
    # result_dict_directory_path = "../../results/prostate_2025-05-26"

    benchmark_dict = {}
    evaluation_dict = {}
    # iterate over the analysed datasets
    for data_display_name, experiment_id_list in experiments_dict.items():
        list_of_evaluated_results, experiment_benchmark_dict = evaluate_repeated_experiments(experiment_id_list, result_dict_directory_path)

        benchmark_dict[data_display_name] = experiment_benchmark_dict
        evaluation_dict[data_display_name] = list_of_evaluated_results


    visualize_runtime_benchmark(benchmark_dict)


evaluate_benchmark_experiment_grid()
