import numpy as np
from sklearn.metrics import r2_score

import utils


def select_features(
    preprocessed_data,
    outer_cv_loop_iteration,
    meta_data,
    reverse_selection_algorithm,
    method,
):
    deselected_features_list = []
    selected_features_dict = {}

    # check if a list of validation train splits or only train data was provided
    try:
        feature_names = preprocessed_data[0][0].columns
    except:  # noqa
        feature_names = preprocessed_data.columns

    # calculate relevance for each feature
    for target_feature_name in feature_names:
        # exclude the label
        if target_feature_name == "label":
            continue

        # get performance of target feature
        (
            performance_evaluation_metric_unlabeled_train_data,
            performance_evaluation_metric_labeled_train_data,
        ) = reverse_selection_algorithm.select_feature_subset(
            preprocessed_data,
            target_feature_name,
            deselected_features_list,
            outer_cv_loop=outer_cv_loop_iteration,
            meta_data=meta_data,
            method=method,
        )

        selected_features_dict[target_feature_name] = (
            performance_evaluation_metric_unlabeled_train_data,
            performance_evaluation_metric_labeled_train_data,
        )
    assert len(selected_features_dict) == len(feature_names) - 1  # exclude the label
    return selected_features_dict


def remove_deselected_features(deselected_features, train_correlation_matrix, train_data_df, test_data_df):
    number_of_initial_features = len(train_correlation_matrix.columns)
    # remove irrelevant features from train_correlation_matrix
    train_correlation_matrix.drop(
        labels=deselected_features,
        axis=0,
        inplace=False,
    )
    train_correlation_matrix.drop(
        labels=deselected_features,
        axis=1,
        inplace=True,
    )
    assert (
        number_of_initial_features - len(deselected_features)
        == train_correlation_matrix.shape[1]
        == train_correlation_matrix.shape[0]
    )
    # remove irrelevant features from test and train data
    train_data_df.drop(columns=deselected_features, inplace=True)
    test_data_df.drop(columns=deselected_features, inplace=True)


def find_correlated_features(train_correlation_matrix, target_feature_name, meta_data):
    # How Correlations Influence Lasso Prediction, April 2012IEEE Transactions on Information Theory 59(3)
    # DOI: 10.1109/TIT.2012.2227680
    # find features correlated to the target_feature from test/ train data
    correlated_features = [
        (feature, correlation_coefficient)
        for feature, correlation_coefficient in train_correlation_matrix[target_feature_name].items()
        if abs(correlation_coefficient) > meta_data["data"]["train_correlation_threshold"]
    ]

    correlated_feature_names = list(map(list, zip(*correlated_features)))[0]

    # check if train data would keep at least one feature after removing label and target_feature
    if train_correlation_matrix.shape[1] - len(correlated_features) < 3:
        absolute_correlations = [
            (feature_name, abs(correlation_coefficient))
            for feature_name, correlation_coefficient in correlated_features
        ]
        # keep the feature with the lowest correlation to the target feature
        sorted_correlated_features = utils.sort_list_of_tuples_by_index(absolute_correlations, index=1, ascending=True)
        min_correlated_feature = sorted_correlated_features[0][0]
        correlated_feature_names.remove(min_correlated_feature)
        assert len(absolute_correlations) - 1 == len(correlated_feature_names)

    assert target_feature_name in correlated_feature_names
    correlated_feature_names.remove(target_feature_name)

    return correlated_feature_names


def calculate_performance_metric_cv(
    params,
    target_feature_name,
    preprocessed_data,
    meta_data,
    method,
    calculate_performance_metric,
    include_label=True,
    deselected_features=None,
):
    performance_metric_list = []

    # cross validation for HPO
    for test_data_df, train_data_df, train_correlation_matrix in preprocessed_data:
        # remove irrelevant features from train_correlation_matrix
        if meta_data["remove_deselected"]:
            remove_deselected_features(
                deselected_features,
                train_correlation_matrix,
                train_data_df,
                test_data_df,
            )

        # remove features correlated to the target feature from train data
        names_of_correlated_features = find_correlated_features(
            train_correlation_matrix, target_feature_name, meta_data
        )

        # remove features correlated to the target_feature from test/ train data and the label if it is not included
        train_df = train_data_df.drop(columns=names_of_correlated_features, inplace=False)
        test_df = test_data_df.drop(columns=names_of_correlated_features, inplace=False)

        if not include_label:
            train_df.drop(columns="label", inplace=True)
            test_df.drop(columns="label", inplace=True)

        model, pruned = calculate_performance_metric(params, test_df, train_df, target_feature_name, method)
        x_test = test_df.drop(columns=target_feature_name).values
        y_test = test_df[target_feature_name].values
        performance_metric_list.append(r2_score(y_test, model.predict(x_test)))

    return np.mean(performance_metric_list)
