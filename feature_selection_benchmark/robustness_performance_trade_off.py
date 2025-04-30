# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Robustness-Performance Trade-Off (RPT) metric calculation module."""

import numpy as np


def rpt(stability, performance, beta=1.0):
    """Calculates the Robustness-Performance Trade-Off metric (rpt).

    A variation on the F-measure (combining precision and recall) to assess
    robustness versus classification performance.

    This function calculates the harmonic mean of model stability and model
    performance, with an adjustable importance weight (beta) to control the
    relative emphasis between the two metrics.

    Args:
        stability : float or array-like
            Stability metric(s) (e.g., Jaccard index), must be in [0, 1].
        performance : float or array-like
            Model performance metric(s) (e.g., accuracy), must be in [0, 1].
        beta : float, optional
            Relative importance of stability versus performance. Default is 1.

    Returns:
        float or np.ndarray
            Harmonic mean(s) of robustness and classification performance.

    References:
        Saeys Y., Abeel T., et al. (2008).
        "Machine Learning and Knowledge Discovery in Databases", pp. 313-325.
        https://link.springer.com/chapter/10.1007/978-3-540-87481-2_21
    """
    stability = np.asarray(stability)
    performance = np.asarray(performance)

    if np.any((stability < 0) | (stability > 1)):
        raise ValueError("Stability must be between 0 and 1.")
    if np.any((performance < 0) | (performance > 1)):
        raise ValueError("Performance must be between 0 and 1.")
    if beta <= 0:
        raise ValueError("Beta must be positive.")

    numerator = ((beta**2) + 1) * stability * performance
    denominator = (beta**2) * stability + performance
    if np.any(denominator == 0):
        raise ValueError("Denominator must not be zero.")

    return numerator / denominator
