# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np


class WeightedManhattanDistance:
    def __init__(self, weights: np.ndarray):
        assert weights.ndim == 1
        self._weights = weights

    def __call__(self, p1, p2):
        assert p1.shape == p2.shape
        assert p1.shape == self._weights.shape
        dist = np.sum(np.abs(p1 - p2) * self._weights)
        return dist
