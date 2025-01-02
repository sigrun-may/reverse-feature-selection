# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Feature selection evaluation test script."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_selection_benchmark.feature_selection_evaluation.evaluate_feature_selection import (
    calculate_performance_metrics_on_shuffled_hold_out_subset,
    train_and_predict,
)

# Create a random number generator
rng = np.random.default_rng()


# Mock data for testing
def create_mock_data(n_samples=50, n_features=200):
    x, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    data = pd.DataFrame(x)
    data.insert(0, "label", y)
    return data


# Mock feature importance
def create_mock_feature_importance():
    return rng.choice([0, 1], size=(200,), p=[0.7, 0.3])


@pytest.fixture
def mock_data():
    return create_mock_data()


@pytest.fixture
def mock_train_data():
    mock = create_mock_data()
    return mock.iloc[:30]


@pytest.fixture
def mock_test_data():
    mock = create_mock_data()
    return mock.iloc[30:]


@pytest.fixture
def mock_feature_importance():
    return create_mock_feature_importance()


@pytest.fixture
def mock_input_result_dict():
    return {
        "reverse_random_forest": [
            {"feature_subset_selection": rng.choice([0, 1], size=(200,), p=[0.7, 0.3])} for _ in range(50)
        ],
        "standard_random_forest": [{"method1": rng.choice([0, 1], size=(200,), p=[0.7, 0.3])} for _ in range(50)],
        "evaluation": {},
    }


def test_train_and_predict(mock_data, mock_feature_importance):
    train_data = mock_data.iloc[:20]
    test_data = mock_data.iloc[20:]
    seed = 42

    predicted_y, predicted_proba_y, y_test = train_and_predict(train_data, test_data, mock_feature_importance, seed)

    assert len(predicted_y) == len(y_test)
    assert len(predicted_proba_y) == len(y_test)
    assert predicted_proba_y.shape[1] == 2  # Check if probabilities for both classes are returned


def test_all_zero_feature_importance(mock_data):
    train_data = mock_data.iloc[:20]
    test_data = mock_data.iloc[20:]
    feature_importance = np.zeros(200)
    with pytest.raises(AssertionError):
        train_and_predict(train_data, test_data, feature_importance, 42)


def test_consistent_random_state(mock_data, mock_feature_importance):
    train_data = mock_data.iloc[:20]
    test_data = mock_data.iloc[20:]
    seed = 42
    y_pred1, y_proba1, y_true1 = train_and_predict(train_data, test_data, mock_feature_importance, seed)
    y_pred2, y_proba2, y_true2 = train_and_predict(train_data, test_data, mock_feature_importance, seed)
    assert np.array_equal(y_pred1, y_pred2)
    assert np.array_equal(y_proba1, y_proba2)


def test_calculate_performance_metrics_with_different_seeds(mock_train_data, mock_feature_importance, mock_test_data):
    seed1 = 2
    seed2 = 424
    seed3 = 20000

    performance_metrics_list1 = calculate_performance_metrics_on_shuffled_hold_out_subset(
        mock_train_data,
        mock_feature_importance,
        mock_test_data,
        seed1,
    )
    performance_metrics_list2 = calculate_performance_metrics_on_shuffled_hold_out_subset(
        mock_train_data,
        mock_feature_importance,
        mock_test_data,
        seed2,
    )
    performance_metrics_list3 = calculate_performance_metrics_on_shuffled_hold_out_subset(
        mock_train_data,
        mock_feature_importance,
        mock_test_data,
        seed3,
    )
    assert performance_metrics_list3 != performance_metrics_list1
    assert performance_metrics_list3 != performance_metrics_list2
    assert performance_metrics_list1 != performance_metrics_list2
    assert all(isinstance(metrics, dict) for metrics in performance_metrics_list1)
    assert all(isinstance(metrics, dict) for metrics in performance_metrics_list2)
    assert all(isinstance(metrics, dict) for metrics in performance_metrics_list3)
    assert all("Accuracy" in metrics for metrics in performance_metrics_list1)
    assert all("Accuracy" in metrics for metrics in performance_metrics_list2)
    assert all("Accuracy" in metrics for metrics in performance_metrics_list3)
    assert all("AUC" in metrics for metrics in performance_metrics_list1)
    assert all("AUC" in metrics for metrics in performance_metrics_list2)
    assert all("AUC" in metrics for metrics in performance_metrics_list3)
    assert all("Average Precision Score" in metrics for metrics in performance_metrics_list1)
    assert all("Average Precision Score" in metrics for metrics in performance_metrics_list2)
    assert all("Average Precision Score" in metrics for metrics in performance_metrics_list3)


def test_calculate_performance_metrics_equal_seeds_for_random_forest(
    mock_train_data, mock_feature_importance, mock_test_data
):
    seed = 42
    performance_metrics_list1 = calculate_performance_metrics_on_shuffled_hold_out_subset(
        mock_train_data, mock_feature_importance, mock_test_data, seed
    )
    performance_metrics_list2 = calculate_performance_metrics_on_shuffled_hold_out_subset(
        mock_train_data, mock_feature_importance, mock_test_data, seed
    )
    assert performance_metrics_list1 == performance_metrics_list2


def test_calculate_performance_metrics_on_shuffled_hold_out_subset(
    mock_train_data, mock_feature_importance, mock_test_data
):
    seed = 42
    performance_metrics_list = calculate_performance_metrics_on_shuffled_hold_out_subset(
        mock_train_data, mock_feature_importance, mock_test_data, seed
    )
    assert isinstance(performance_metrics_list, list)
    for metrics in performance_metrics_list:
        assert isinstance(metrics, dict)
        assert "Accuracy" in metrics
        assert "AUC" in metrics
        assert "Average Precision Score" in metrics
