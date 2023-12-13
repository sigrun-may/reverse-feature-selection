import numpy as np


class WeightedManhattanDistance:
    def __init__(self, weights: np.ndarray):
        assert weights.ndim == 1
        self._weights = weights

    def __call__(self, p1, p2):
        dist = np.sum(np.abs(p1 - p2) * self._weights)
        return dist
