# coding: utf-8
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import PowerTransformer, minmax_scale
from sklearn.metrics.pairwise import manhattan_distances


import warnings

warnings.filterwarnings("ignore")


def validate(
    test: pd.DataFrame,
    train: pd.DataFrame,
    # selected_feature_subset,
    selected_feature_subset: Dict[str, float],
    number_of_neighbors: int,
) -> Tuple[list[int], list[int]]:

    # test_data = test[selected_feature_subset.keys()].values
    # train_data = train[selected_feature_subset.keys()].values
    # assert test_data.size == len(selected_feature_subset.keys())

    test_data = test[selected_feature_subset]
    train_data = train[selected_feature_subset]
    # assert test_data.size == len(selected_feature_subset)

    # only use the labels that appear in the data
    list_of_classes = np.unique(train["label"])
    # number_of_samples = train_data.shape[0]

    # z transform to compare the distances
    powerTransformer = PowerTransformer(
        copy=True, method="yeo-johnson", standardize=True
    )
    scaled_train_data = powerTransformer.fit_transform(train_data)
    scaled_test_data = powerTransformer.transform(test_data)

    assert (
        scaled_train_data.shape[1]
        == train_data.shape[1]
        == scaled_test_data.shape[1]
        == test_data.shape[1]
    )
    assert scaled_train_data.shape[0] == train_data.shape[0]

    # man = manhattan_distances(scaled_test_data, scaled_train_data, sum_over_features = False)

    weights = np.asarray(
        [
            selected_feature_subset[selected_feature]
            for selected_feature in train_data.columns
        ]
    )
    assert type(weights) == np.ndarray, type(weights)
    assert type(weights[0]) == np.float64, weights

    classified_classes = []
    for i in range(scaled_test_data.shape[0]):
        test_sample = scaled_test_data[i, :]

        distance_to_each_class = []
        for class_number in list_of_classes:
            class_sample_mask = train["label"].to_numpy() == class_number
            class_samples = scaled_train_data[class_sample_mask]

            weighted_distances = []
            for class_sample in class_samples:
                # calculate weighted manhattan distance
                weighted_distance = 0
                for j in range(class_sample.size):
                    weighted_distance += (
                        np.abs(class_sample[j] - test_sample[j]) * weights[j]
                    )

                # distance = class_sample - test_sample
                # absolute_distance = np.absolute(distance)
                # man = manhattan_distances(class_sample, test_sample)
                # assert man == absolute_distance
                # # weights = minmax_scale(weights)
                # weighted_distance = absolute_distance  # * weights
                weighted_distances.append(weighted_distance)
                # weighted_distances.append(np.sum(np.absolute(class_sample - test_sample) * weights))

            nearest_neighbors = []
            for neighbor in range(number_of_neighbors):
                min_index = np.argmin(weighted_distances)
                nearest_neighbors.append(weighted_distances.pop(min_index))
            distance_to_each_class.append(np.sum(nearest_neighbors))

        # print("distance to classes:", distance_to_each_class)

        # the index of distance_to_each_class equals the integer representation of the class
        classified_class = distance_to_each_class.index(min(distance_to_each_class))
        classified_classes.append(classified_class)

        # classified_classes.append(classified_class)
        #
        # classification_result[selected_sample] = (
        #     classified_class == train.iloc[selected_sample, 0]
        # )
        #
        # # classes = np.int_([1, 0])
        # # true_classes.astype("int")
        # # predicted_classes = np.asarray(classified_classes)
        # # predicted_classes.astype(int)
        # # Only use the labels that appear in the data
        # # classes = unique_labels(true_classes, predicted_classes)
        # # plotter.plot_confusion_matrix(true_classes, predicted_classes, classes, True, title=None, cmap=pyplot.cm.Blues)
        # # pyplot.savefig(settings['EXPERIMENT_NAME'] + '.png')
        # # pyplot.show()
        # print("##")
        # print(classified_classes == true_classes)
        # print("##")

        # TODO return only classified classes to print or plot results?
    assert len(classified_classes) == scaled_test_data.shape[0]
    true_classes = list(test["label"])
    return classified_classes, true_classes


def validate_standard(
    test: pd.DataFrame,
    train: pd.DataFrame,
    # selected_feature_subset,
    selected_feature_subset: Dict[str, float],
    number_of_neighbors: int,
) -> Tuple[list[int], list[int]]:

    # test_data = test[selected_feature_subset.keys()].values
    # train_data = train[selected_feature_subset.keys()].values
    # assert test_data.size == len(selected_feature_subset.keys())

    test_data = test[selected_feature_subset]
    train_data = train[selected_feature_subset]
    # assert test_data.size == len(selected_feature_subset)

    # only use the labels that appear in the data
    list_of_classes = np.unique(train["label"])
    # number_of_samples = train_data.shape[0]

    # z transform to compare the distances
    powerTransformer = PowerTransformer(
        copy=True, method="yeo-johnson", standardize=True
    )
    scaled_train_data = powerTransformer.fit_transform(train_data)
    scaled_test_data = powerTransformer.transform(test_data)

    assert (
        scaled_train_data.shape[1]
        == train_data.shape[1]
        == scaled_test_data.shape[1]
        == test_data.shape[1]
    )
    assert scaled_train_data.shape[0] == train_data.shape[0]

    weights = np.asarray(
        [
            selected_feature_subset[selected_feature]
            for selected_feature in train_data.columns
        ]
    )
    assert type(weights) == np.ndarray, type(weights)
    assert type(weights[0]) == np.float64, weights

    classified_classes = []
    for scaled_test_sample in scaled_test_data:
        assert scaled_test_sample.size == scaled_test_data.shape[1]
        # assert np.array_equal(scaled_test_sample, scaled_test_data[0, :])
        labeled_distances_list = _get_distances(
            scaled_test_sample, scaled_train_data, train["label"], weights
        )
        classified_classes.append(
            _classify(labeled_distances_list, number_of_neighbors, list_of_classes)
        )

        # TODO return only classified classes to print or plot results?
    assert len(classified_classes) == scaled_test_data.shape[0]
    true_classes = list(test["label"])
    return classified_classes, true_classes


def _classify(labeled_distances_list, number_of_neighbors, list_of_classes):
    ascending_labeled_distances_list = sort_list_of_tuples_by_first_value(
        labeled_distances_list
    )
    nearest_neighbors = ascending_labeled_distances_list[:number_of_neighbors]

    classes = []
    for label, distance in nearest_neighbors:
        classes.append(label)

    class_counts = []
    for class_label in list_of_classes:
        class_counts.append(sum(x == class_label for x in classes))

    max_index = np.argmax(class_counts)
    predicted_class = list_of_classes[max_index]
    return predicted_class


def sort_list_of_tuples_by_first_value(list_to_be_sorted):
    # getting length of list of tuples
    lst = len(list_to_be_sorted)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if list_to_be_sorted[j][1] > list_to_be_sorted[j + 1][1]:
                temp = list_to_be_sorted[j]
                list_to_be_sorted[j] = list_to_be_sorted[j + 1]
                list_to_be_sorted[j + 1] = temp
    return list_to_be_sorted


def _get_distances(scaled_test_sample, scaled_train_data, train_labels, weights):
    assert scaled_test_sample.size == weights.size

    labeled_weighted_distances = []
    for scaled_train_sample, label in zip(scaled_train_data, train_labels):
        assert scaled_test_sample.shape == scaled_train_sample.shape
        # calculate weighted manhattan distance
        distances = np.abs(scaled_train_sample - scaled_test_sample)
        weighted_distances = distances * weights
        distance = np.sum(weighted_distances)

        labeled_weighted_distances.append((label, distance))
    assert len(labeled_weighted_distances) == scaled_train_data.shape[0]
    return labeled_weighted_distances
