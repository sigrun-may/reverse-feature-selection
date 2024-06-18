# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""This module evaluates the feature selection results of different feature selection methods."""

import pickle
from pathlib import Path

import numpy as np

from feature_selection_benchmark.feature_selection_evaluation import stability_estimator


result_dict_path = Path(
    f"/home/sigrun/PycharmProjects/reverse_feature_selection/results/colon_standard_rf_result_dict.pkl"
)
if result_dict_path.exists():
    with open(result_dict_path, "rb") as file:
        result_dict = pickle.load(file)

result_dict["reverse_random_forest"] = []
feature_importance_matrix = np.zeros((30, 2000))

reverse_feature_selection_result = result_dict["reverse_random_forest"]
standard_feature_selection_result = result_dict["standard_random_forest"]

# # initialize feature importance matrix with zeros
# feature_importance_matrix = np.zeros(
#     (len(reverse_feature_selection_result), len(reverse_feature_selection_result[0]))
# )
# # for i, cv_iteration_result in enumerate(reverse_feature_selection_result):
#     # check if the value in the p_value column of the DataFrame is greater than 0.05 and at the same time the mean of the
#     # labeled OOB error is smaller than the mean of the unlabeled OOB error
#
#     # boolean_index_array = cv_iteration_result["p_value"] < 0.05 and np.mean(cv_iteration_result["labeled_errors"]) < np.mean(cv_iteration_result["unlabeled_errors"])
#
#     selected_feature_subset_index = []
#     selected_feature_subset = []
#     # iterate over the rows of the DataFrame
#     for index, labeled_error, unlabeled_error, fraction_mean, fraction_median, p_value in zip(cv_iteration_result.index, cv_iteration_result["labeled_errors"], cv_iteration_result["unlabeled_errors"], cv_iteration_result["fraction_mean"], cv_iteration_result["fraction_median"], cv_iteration_result["p_value"]):
#         # generate boolean index array for the condition
#         if p_value > 0.05 and np.mean(labeled_error) < np.mean(unlabeled_error):
#             selected_feature_subset_index.append(index)
#             selected_feature_subset.append(fraction_mean)
#         else:
#             selected_feature_subset.append(0)
#     feature_importance_matrix[i, :] = np.asarray(selected_feature_subset)
# print(stability_estimator.get_stability(feature_importance_matrix))

feature_importance_matrix_gini = np.zeros_like(feature_importance_matrix)
feature_importance_matrix_shap = np.zeros_like(feature_importance_matrix)
feature_importance_matrix_permutation = np.zeros_like(feature_importance_matrix)
feature_importance_matrix_gini_unoptimized = np.zeros_like(feature_importance_matrix)

for i, cv_iteration_result in enumerate(standard_feature_selection_result):
    feature_importance_matrix_gini[i, :] = cv_iteration_result["gini_importance"]
    feature_importance_matrix_permutation[i, :] = cv_iteration_result["permutation_importance"]
    print("perm", np.sum(cv_iteration_result["permutation_importance"] > 0))
    feature_importance_matrix_gini_unoptimized[i, :] = cv_iteration_result["gini_rf_unoptimized_importance"]
    feature_importance_matrix_shap[i, :] = cv_iteration_result["summed_shap_values"]
print("gini", stability_estimator.get_stability(feature_importance_matrix_gini))
print("perm", stability_estimator.get_stability(feature_importance_matrix_permutation))
print("gini_rf_unoptimized", stability_estimator.get_stability(feature_importance_matrix_gini_unoptimized))
print("shap", stability_estimator.get_stability(feature_importance_matrix_shap))
