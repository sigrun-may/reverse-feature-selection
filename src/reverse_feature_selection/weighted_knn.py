# coding: utf-8
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import PowerTransformer, minmax_scale

from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    mutual_info_score,
)
import warnings

warnings.filterwarnings("ignore")


def validate_feature_subset(
    test: pd.DataFrame,
    train: pd.DataFrame,
    selected_feature_subset: Dict[str, float],
    number_of_neighbors: int,
) -> Dict[str, float]:

    test_data = test[selected_feature_subset.keys()].values
    train_data = train[selected_feature_subset.keys()].values

    list_of_classes = np.unique(train_data[:, 0])
    number_of_samples = test_data.shape[0]

    # initialize the classification results array with False
    classification_result = np.full(number_of_samples, False)
    classified_classes = []

    # the classes equal the labels as integer representation
    # Only use the labels that appear in the data
    classes = unique_labels(train["label"])  # TODO check classes to int
    print(classes)
    true_classes = test_data[:, 0]
    true_classes.astype(int)

    unlabeled_train_data = train_data[:, 1:]
    unlabeled_test_data = test_data[:, 1:]

    # z transform to compare the distances
    powerTransformer = PowerTransformer(
        copy=True, method="yeo-johnson", standardize=True
    )
    scaled_train_data = powerTransformer.fit_transform(unlabeled_train_data)
    scaled_test_data = powerTransformer.transform(unlabeled_test_data)

    for selected_sample in range(number_of_samples):
        test_sample = scaled_test_data[selected_sample]

        distance_to_each_class = []

        for class_number in list_of_classes:
            all_sample_indices_from_class = train_data[:, 0] == class_number
            all_model_samples_from_one_class = scaled_train_data[
                all_sample_indices_from_class
            ]

            weighted_distances = []
            weights = selected_feature_subset.values()
            for class_sample in all_model_samples_from_one_class:
                # calculate weighted manhattan distance
                distance = class_sample - test_sample
                absolute_distance = np.absolute(distance)
                # scaled_weights = minmax_scale(weights)
                weighted_distance = absolute_distance * weights
                weighted_distances.append(np.sum(weighted_distance))
                # weighted_distances.append(np.sum(np.absolute(class_sample - test_sample) * weights))

            # TODO trimmed mean oder median
            nearest_neighbors = []
            for neighbor in range(number_of_neighbors):
                min_index = np.argmin(weighted_distances)[0]
                nearest_neighbors.append(weighted_distances.pop(min_index))
            distance_to_each_class.append(np.sum(nearest_neighbors))

        print("distance to classes:", distance_to_each_class)
        minimum_distance = min(distance_to_each_class)

        # the index of distance_to_each_class equals the integer representation of the class
        classified_class = distance_to_each_class.index(minimum_distance)
        print("classified class:", classified_class)
        print("true class:", test_data[selected_sample, 0])

        classified_classes.append(classified_class)

        classification_result[selected_sample] = (
            classified_class == test_data[selected_sample, 0]
        )

    # classes = np.int_([1, 0])
    true_classes.astype("int")
    predicted_classes = np.asarray(classified_classes)
    predicted_classes.astype(int)
    # Only use the labels that appear in the data
    classes = unique_labels(true_classes, predicted_classes)
    # plotter.plot_confusion_matrix(true_classes, predicted_classes, classes, True, title=None, cmap=pyplot.cm.Blues)
    # pyplot.savefig(settings['EXPERIMENT_NAME'] + '.png')
    # pyplot.show()
    print("##")
    print(classified_classes == true_classes)
    print("##")
    metrics_dict = {
        "matthews": matthews_corrcoef(true_classes, classified_classes),
        "accuracy": accuracy_score(true_classes, classified_classes),
        "f1_score": f1_score(true_classes, classified_classes),
        "balanced_accuracy_score": balanced_accuracy_score(
            true_classes, classified_classes
        ),
        "mutual_info_score": mutual_info_score(true_classes, classified_classes),
    }
    # TODO return only classified classes to print or plot results?
    return metrics_dict
