from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score

from src.reverse_feature_selection import weighted_knn
from src.reverse_feature_selection.misc import evaluate_selected_features

import settings


def extract_indices_and_results(feature_selection_result):
    # rewrite data structure
    result_dict = {}
    test_train_indices = []
    for feature_selection_dict, test_indices, train_indices in feature_selection_result:
        test_train_indices.append((test_indices, train_indices))
        for selection_method, selected_features in feature_selection_dict.items():
            selected_feature_subset_list = result_dict.get(selection_method, [])
            selected_feature_subset_list.append(selected_features)
            result_dict[selection_method] = selected_feature_subset_list
    assert len(test_train_indices) == settings.N_FOLDS_OUTER_CV
    return result_dict, test_train_indices


def calculate_weights_for_reverse_lasso(selected_feature_subsets, delta=0, factor=1):
    weighted_selected_feature_subsets = []
    for feature_subset in selected_feature_subsets:
        subset = {}
        for feature in feature_subset.keys():
            if (
                feature_subset[feature][1]
                >= feature_subset[feature][0] + delta * factor
            ):
                subset[feature] = (
                    feature_subset[feature][1] - feature_subset[feature][0]
                )
        weighted_selected_feature_subsets.append(subset)
    return weighted_selected_feature_subsets


def extract_data_structure_from_intersect(selected_feature_subsets, intersect):
    intersect_dict_list = []
    for subset_dict in selected_feature_subsets:
        intersect_dict = dict()
        for key, value in subset_dict.items():
            if key in intersect:
                intersect_dict[key] = value
        intersect_dict_list.append(intersect_dict)
    return intersect_dict_list


def validate_feature_subset(
    k_neighbors, test_train_indices_list, feature_subset_dict_list, data_df
):
    assert len(feature_subset_dict_list) == len(test_train_indices_list)
    assert isinstance(feature_subset_dict_list, list)
    assert isinstance(test_train_indices_list, list)
    results = []
    for outer_cv_loop_iteration, (
        test_indices,
        train_indices,
    ) in enumerate(test_train_indices_list):
        results.append(
            weighted_knn.validate(
                data_df.iloc[test_indices, :],
                data_df.iloc[train_indices, :],
                # intersect,
                feature_subset_dict_list[outer_cv_loop_iteration],
                k_neighbors,
            )
        )
    assert len(results) == len(test_train_indices_list)

    # flatten lists of results from k-fold cross-validation
    predicted_classes = []
    true_classes = []
    for predicted_classes_sublist, true_classes_sublist in results:
        predicted_classes.extend(predicted_classes_sublist)
        true_classes.extend(true_classes_sublist)

    return predicted_classes, true_classes


def calculate_metrics(predicted_classes, true_classes):
    print(
        "matthews",
        matthews_corrcoef(true_classes, predicted_classes),
        "accuracy",
        accuracy_score(true_classes, predicted_classes),
        "f1_score",
        f1_score(true_classes, predicted_classes),
        "balanced_accuracy_score",
        balanced_accuracy_score(true_classes, predicted_classes),
    )
    return {
        "matthews": matthews_corrcoef(true_classes, predicted_classes),
        "accuracy": accuracy_score(true_classes, predicted_classes),
        "f1_score": f1_score(true_classes, predicted_classes),
        "balanced_accuracy_score": balanced_accuracy_score(
            true_classes, predicted_classes
        ),
    }


def validate_feature_subsets(
    feature_selection_result_dict,
    test_train_indices_list,
    data_df,
    k_neighbors,
    delta=0,
    factor=1,
):
    metrics_per_method_dict = {}
    for (
        feature_selection_method,
        selected_feature_subsets,
    ) in feature_selection_result_dict.items():
        print("##############################", feature_selection_method, ": ")
        union, intersect = evaluate_selected_features(selected_feature_subsets)

        if feature_selection_method == "reverse_lasso":
            selected_feature_subsets = calculate_weights_for_reverse_lasso(
                selected_feature_subsets, delta, factor
            )

            union, intersect = evaluate_selected_features(selected_feature_subsets)
            # number_of_features.append(len(intersect))

        intersect_dict_list = extract_data_structure_from_intersect(
            selected_feature_subsets, intersect
        )
        predicted_classes, true_classes = validate_feature_subset(
            k_neighbors, test_train_indices_list, intersect_dict_list, data_df
        )
        metrics_dict = calculate_metrics(predicted_classes, true_classes)
        metrics_per_method_dict[feature_selection_method] = metrics_dict
    return metrics_per_method_dict
