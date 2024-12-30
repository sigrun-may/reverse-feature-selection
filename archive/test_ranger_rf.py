# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Test for ranger random forest feature importance calculation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_selection_benchmark.ranger_rf import (
    calculate_feature_importance,
    optimized_ranger_random_forest_importance,
    ranger_random_forest,
    sklearn_random_forest,
)


# Mock data for testing
def create_mock_data(n_samples=30, n_features=200):
    x, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    data = pd.DataFrame(x)
    data.insert(0, "label", y)
    return data


@pytest.fixture
def sample_data():
    return create_mock_data()


@pytest.fixture
def train_indices():
    return np.asarray(range(29))


@pytest.fixture
def meta_data():
    return {"random_state": 42, "max_trees_random_forest": 100, "n_trials_optuna": 10, "verbose_optuna": False}


@patch("feature_selection_benchmark.ranger_rf.ranger_random_forest")
@patch("optuna.create_study")
def test_optimized_ranger_random_forest_importance(
    mock_create_study, mock_ranger_rf, sample_data, train_indices, meta_data
):
    mock_study = MagicMock()
    mock_study.best_params = {"max_depth": 10, "num_trees": 50, "mtry": 2, "regularization_factor": 0.1}
    mock_create_study.return_value = mock_study
    mock_ranger_rf.return_value = (0.1, np.random.rand(sample_data.shape[1] - 1))

    result = optimized_ranger_random_forest_importance(sample_data, train_indices, meta_data)
    assert "permutation" in result
    assert "best_params_ranger_permutation" in result
    assert "oob_score_permutation" in result


@patch("concurrent.futures.ProcessPoolExecutor")
def test_calculate_feature_importance(mock_executor, sample_data, train_indices, meta_data):
    mock_future = MagicMock()
    mock_future.result.side_effect = [
        {"gini_impurity": np.random.rand(sample_data.shape[1] - 1)},
        {"permutation": np.random.rand(sample_data.shape[1] - 1)},
    ]
    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

    result = calculate_feature_importance(sample_data, train_indices, meta_data)
    assert "gini_impurity" in result
    assert "permutation" in result


@patch("optuna.create_study")
def test_sklearn_random_forest(mock_create_study, sample_data, train_indices, meta_data):
    mock_study = MagicMock()
    mock_study.best_params = {
        "max_depth": 10,
        "n_estimators": 50,
        "max_features": 2,
        "min_samples_leaf": 3,
        "min_samples_split": 2,
        "min_impurity_decrease": 0.1,
    }
    mock_create_study.return_value = mock_study

    result = sklearn_random_forest(sample_data, train_indices, meta_data)
    assert "gini_impurity" in result
    assert "train_oob_score_sklearn" in result
    assert "best_params_sklearn" in result


def test_ranger_random_forest():
    sample_data = create_mock_data()
    train_indices = np.asarray(range(29))
    hyperparameters = {"max_depth": 10, "num_trees": 50, "mtry": 2, "seed": 42, "regularization_factor": 0.1}
    result = ranger_random_forest(sample_data, train_indices, hyperparameters)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], np.ndarray)
    assert len(result[1]) == sample_data.shape[1] - 1  # exclude the label column
    assert all(isinstance(x, float) for x in result[1]), "Result vector contains non-float values."
