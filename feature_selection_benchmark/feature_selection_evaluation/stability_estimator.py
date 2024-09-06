# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Stability estimator for feature selection in high-dimensional data with small sample size.

A generalized stability estimator based on inter-intrastability of subsets for high-dimensional feature selection.
https://doi.org/10.1016/j.chemolab.2021.104457
"""
import numpy as np


def calculate_stability(selected_features_matrix, flip=False) -> float:
    """Calculate the stability of selected features.

    Args:
        selected_features_matrix: Numpy array containing the selected features.
            Rows represent the folds, columns represent the features.
            The matrix should contain binary values (0 or 1).
        flip: Boolean flag to flip 0 and 1 in the selected features matrix. Avoids division by zero. Default is False.

    Returns:
        Stability of selected features.
    """
    # preserve original matrix
    selected_features_matrix_cp = selected_features_matrix.copy()

    # check if matrix is empty
    if selected_features_matrix_cp.size == 0:
        raise ValueError("Empty matrix for stability estimation")

    # check if matrix only contains zeros
    if np.all(selected_features_matrix_cp == 0):
        print("Only zeros in matrix for stability estimation: no features selected")
        return 0

    # check if matrix is already binary
    if not np.array_equal(np.unique(selected_features_matrix_cp), np.array([0, 1])):
        # binarize matrix and set all values not equal to zero to one
        selected_features_matrix_cp = np.where(selected_features_matrix != 0, 1, 0)

    # check if matrix only contains ones
    if np.all(selected_features_matrix_cp == 1):
        print("Only ones in matrix for stability estimation: all features selected")
        return 1

    if flip:
        # flip 0 and 1 for the feature importance matrix
        selected_features_matrix_cp = 1 - selected_features_matrix_cp

    assert np.array_equal(np.unique(selected_features_matrix_cp), np.array([0, 1])), np.unique(
        selected_features_matrix_cp
    )

    robustness_vector = selected_features_matrix_cp.sum(axis=0)
    subset_vector = selected_features_matrix_cp.sum(axis=1)

    number_of_features = len(robustness_vector)
    number_of_folds = len(subset_vector)

    stability = 0.0
    count_k = 0
    for k in range(1, number_of_folds + 1):
        count_k += 1

        # empirical density of the robustness_vector
        # number of features which were selected k-times
        robustness_density = list(robustness_vector).count(k)
        subset_size_stability = _subset_size_stability(subset_vector, number_of_features, k)

        stability += (k**2 * robustness_density * subset_size_stability) / subset_vector[k - 1]
        if np.isnan(stability):
            print(f"stability is nan - k: {k}, robustness_density: {robustness_density}, subset_size_stability: {subset_size_stability}, subset_vector[k - 1]: {subset_vector[k - 1]}")
            return 0

    if stability > number_of_folds**2:
        print(stability)
    stability = stability / (number_of_folds**2)

    assert count_k == number_of_folds
    if stability > 1:
        print(stability, " stability greater than 1")
    # assert stability <= 1.1, stability
    return stability


def _subset_size_stability(subset_vector, number_of_features, k):
    """Calculate the subset-size stability.

    Args:
        subset_vector: Numpy array containing the number of selected features
            per fold. The length of the array equals the number of all folds.
        number_of_features: Number of all features in the input data.
        k: Subset size.

    Returns:
        Subset-size stability
    """
    k = k - 1  # Shift from one-based k to zero-based for correct indexing
    assert k >= 0
    if k == 0:
        subset_size_stability = subset_vector[k] / number_of_features

    elif subset_vector[k] > subset_vector[k - 1]:
        subset_size_stability = subset_vector[k - 1] / subset_vector[k]

    elif subset_vector[k] < subset_vector[k - 1]:
        subset_size_stability = subset_vector[k] / subset_vector[k - 1]

    elif subset_vector[k] == subset_vector[k - 1]:
        subset_size_stability = 1

    else:
        raise ValueError("Incorrect subset vector")

    return subset_size_stability

