from typing import List, Dict, Tuple
import numpy as np


def get_stability(used_features_matrix):
    # assert len(subset_vector) == number_of_folds  # lambda
    # assert (
    #     len(robustness_vector) == number_of_features
    # ), f"{len(robustness_vector)}, {number_of_features}"
    robustness_vector = used_features_matrix.sum(axis = 0)
    subset_vector = used_features_matrix.sum(axis = 1)

    number_of_features = len(robustness_vector)
    number_of_folds = len(subset_vector)

    stability = 0.0
    count_k = 0
    for k in range(1, number_of_folds + 1):
        count_k += 1

        stability += (
            k**2
            # empirical density of the robustness_vector
            # number of features which were selected k-times
            * list(robustness_vector).count(k)
            * _subset_size_stability(subset_vector, number_of_features, k)
        ) / subset_vector[k - 1]
        if np.isnan(stability):
            print('stability is nan')
            return 0

    stability = stability / (number_of_folds**2)

    assert count_k == number_of_folds
    return stability


def _subset_size_stability(subset_vector, number_of_features, k):
    """

    Args:
        subset_vector: Numpy array containing the number of selected features
            per fold. The length of the array equals the number of all folds.
        number_of_features: Number of all features in the input data.

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
