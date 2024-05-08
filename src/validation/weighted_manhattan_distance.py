# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Feature weighted Manhattan distance."""

import numpy as np


class WeightedManhattanDistance:
    """Calculate the weighted Manhattan distance between two points."""

    def __init__(self, weights: np.ndarray):
        """
        Initialize the weighted Manhattan distance.

        Args:
            weights: The weights for the Manhattan distance calculation.

        Raises:
            AssertionError: If the weights are not one-dimensional.
        """
        assert weights.ndim == 1
        self._weights = weights

    def __call__(self, p1, p2):
        """
        Calculate the weighted Manhattan distance between two points.

        Args:
            p1: The first point.
            p2: The second point.

        Returns:
            The weighted Manhattan distance between the two points.

        Raises:
            AssertionError: If the shapes of the points and weights do not match.
        """
        assert p1.shape == p2.shape
        assert p1.shape == self._weights.shape
        dist = np.sum(np.abs(p1 - p2) * self._weights)
        return dist
