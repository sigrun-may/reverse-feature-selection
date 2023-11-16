import numpy as np
import pandas as pd
from numpy import ravel
from sklearn.ensemble import RandomForestRegressor

from src.reverse_rf import preprocessing


def calculate_validation_metric_per_feature(
    train_index, validation_index, data_df, meta_data
):
    # preprocess data
    data_split = preprocessing.preprocess_data(
        train_index, validation_index, data_df, correlation_matrix=True
    )

    scores_labeled_list = []
    scores_unlabeled_list = []

    clf = RandomForestRegressor(
        warm_start=False,
        max_features=None,
        oob_score=True,
        # max_depth=10,
        # n_estimators=30,
        criterion="absolute_error",
        random_state=42,
        min_samples_leaf=2,
    )

    # iterate over all features
    for target_feature_name in data_df.columns[1:]:
        (
            x_train,
            y_train,
            x_validation,
            y_validation,
        ) = preprocessing.get_uncorrelated_train_and_validation_data(
            data_split=data_split,
            target_feature=target_feature_name,
            labeled=True,
            meta_data=meta_data,
        )
        # predict the target feature including the label in the training data
        clf.fit(x_train, ravel(y_train))
        score_labeled = clf.oob_score_
        label_zero = clf.feature_importances_[0] == 0
        assert clf.feature_importances_.size == x_train.shape[1]

        # preprocess data without label
        preprocessed_data = preprocessing.get_uncorrelated_train_and_validation_data(
            data_split=data_split,
            target_feature=target_feature_name,
            labeled=False,
            meta_data=meta_data,
        )
        if preprocessed_data is not None:
            x_train, y_train, x_validation, y_validation = preprocessed_data

            # predict the target feature without the label information
            clf.fit(x_train, ravel(y_train))
            score_unlabeled = clf.oob_score_
        # no feature was uncorrelated to the target feature
        else:
            score_unlabeled = 0

        if (
            label_zero # label was not selected
            or score_labeled <= score_unlabeled  # labeled training was not better than training without label
            or (
                score_unlabeled <= 0
                and score_labeled <= 0
                and abs(score_labeled - score_unlabeled) < 0.4  # both scores are negative,
                                                                # but have a significant distance
            )
        ):
            score_unlabeled = score_labeled = 0
        else:
            print(
                f"{target_feature_name} selected {abs(score_unlabeled-score_labeled)}, "
                f"ul{score_unlabeled}, l{score_labeled}"
            )
        scores_labeled_list.append(score_labeled)
        scores_unlabeled_list.append(score_unlabeled)

    assert (
        len(scores_labeled_list) == len(scores_unlabeled_list) == data_df.shape[1] - 1
    )
    result_df = pd.DataFrame(
        data=scores_unlabeled_list, index=data_df.columns[1:], columns=["unlabeled"]
    )
    result_df["labeled"] = scores_labeled_list
    result_df["distance"] = abs(
        np.asarray(scores_labeled_list) - np.asarray(scores_unlabeled_list)
    )
    return result_df
