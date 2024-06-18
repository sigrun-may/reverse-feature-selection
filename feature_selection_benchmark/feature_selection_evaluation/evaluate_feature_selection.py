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


result_dict_path = Path("../../results/colon_standard_rf_result_dict.pkl")
if result_dict_path.exists():
    with open(result_dict_path, "rb") as file:
        result_dict = pickle.load(file)

result_dict["reverse_random_forest"] = []  # TODO remove this line
reverse_feature_selection_result = result_dict["reverse_random_forest"]
standard_feature_selection_result = result_dict["standard_random_forest"]


# # initialize feature importance matrix with zeros
# feature_importance_matrix = np.zeros(
#     (len(reverse_feature_selection_result), len(reverse_feature_selection_result[0]))
# )
feature_importance_matrix = np.zeros((30, 2000))  # TODO remove this line
feature_importance_matrix_gini = np.zeros_like(feature_importance_matrix)
feature_importance_matrix_shap = np.zeros_like(feature_importance_matrix)
feature_importance_matrix_permutation = np.zeros_like(feature_importance_matrix)
feature_importance_matrix_gini_unoptimized = np.zeros_like(feature_importance_matrix)

for i, cv_iteration_result in enumerate(standard_feature_selection_result):
    feature_importance_matrix_gini[i, :] = cv_iteration_result["gini_importance"]
    feature_importance_matrix_permutation[i, :] = cv_iteration_result["permutation_importance"]
    feature_importance_matrix_gini_unoptimized[i, :] = cv_iteration_result["gini_rf_unoptimized_importance"]
    feature_importance_matrix_shap[i, :] = cv_iteration_result["summed_shap_values"]
print("gini", stability_estimator.get_stability(feature_importance_matrix_gini))
print("perm", stability_estimator.get_stability(feature_importance_matrix_permutation))
print("gini_rf_unoptimized", stability_estimator.get_stability(feature_importance_matrix_gini_unoptimized))
print("shap", stability_estimator.get_stability(feature_importance_matrix_shap))
