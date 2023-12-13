import multiprocessing
import pickle
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import ravel
from scipy.stats import ttest_ind
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
)

from src.reverse_feature_selection import preprocessing


def calculate_oob_scores(x_train, y_train, meta_data):
    # calculate oob_scores for random forest regressors with different random seeds

    oob_scores_labeled = []
    oob_scores_unlabeled = []

    # Perform validation using different random seeds
    for i in range(5):
        # Create a RandomForestRegressor model with specified parameters
        # TODO move parameter setting to settings.toml
        clf1 = RandomForestRegressor(
            warm_start=False,
            max_features=None,
            oob_score=mean_squared_error,  # Use out-of-bag score for evaluation
            n_estimators=100,
            # random_state=seed,
            min_samples_leaf=2,
        )
        # Create a copy of the RandomForestRegressor (clf1)
        clf2 = clone(clf1)

        # Fit the first model with training data including the label
        assert "label" in x_train.columns
        clf1.fit(x_train, y_train)
        label_importance_zero = clf1.feature_importances_[0] == 0

        # If the feature importance of the label feature is zero, it means the label was not considered in the model
        if label_importance_zero:
            return None, None

        # Store the OOB score for the labeled model
        oob_scores_labeled.append(clf1.oob_score_)

        # Fit the second model to the unlabeled training data (excluding 'label' column)
        unlabeled_x_train = x_train.loc[:, x_train.columns != "label"]
        assert unlabeled_x_train.shape[1] == x_train.shape[1] - 1
        assert "label" not in unlabeled_x_train.columns
        clf2.fit(unlabeled_x_train, y_train)

        # Store the OOB score for the unlabeled model
        oob_scores_unlabeled.append(clf2.oob_score_)

    # print("oob_scores_labeled", oob_scores_labeled, "oob_scores_unlabeled", oob_scores_unlabeled)
    return oob_scores_labeled, oob_scores_unlabeled


def calculate_mean_oob_scores_and_p_value(target_feature_name, outer_cv_loop, meta_data):
    # calculate the mean oob_scores for random forest regressors with different random seeds
    # for training data including the label and without the label for the given target feature

    mean_oob_score_labeled = 0
    mean_oob_score_unlabeled = 0
    p_value = None

    pickle_base_path = Path(
        f"../../preprocessed_data/{meta_data['data']['name']}/outer_fold_{outer_cv_loop}"
    )
    assert pickle_base_path.exists(), f"{pickle_base_path} does not exist"

    # Load the cached preprocessed data for the given outer cross-validation fold
    with open(f"{pickle_base_path}/train.pkl", "rb") as file:
        train_df = pickle.load(file)
        assert target_feature_name in train_df.columns
        assert "label" in train_df.columns
    with open(f"{pickle_base_path}/train_correlation_matrix.pkl", "rb") as file:
        corr_matrix_df = pickle.load(file)
        assert "label" not in corr_matrix_df.columns
    assert train_df.shape[1] - 1 == corr_matrix_df.shape[0]  # corr_matrix_df does not include the label

    # Prepare training data
    y_train = ravel(train_df[target_feature_name])

    # Remove features correlated to the target feature
    x_train = preprocessing.remove_features_correlated_to_target_feature(
        train_df, corr_matrix_df, target_feature_name, meta_data
    )
    if x_train is None:
        return None, None, None

    assert target_feature_name not in x_train.columns

    # Calculate out-of-bag (OOB) scores for labeled and unlabeled training data
    oob_scores_labeled, oob_scores_unlabeled = calculate_oob_scores(x_train, y_train, meta_data)

    # Check if OOB scores for labeled data are available and if training with the label is better than without the label
    if oob_scores_labeled is not None and abs(np.mean(oob_scores_labeled)) < abs(np.mean(oob_scores_unlabeled)):
        # Calculate the percentage difference between OOB scores
        absolute_percentage_difference = (
            (np.mean(oob_scores_unlabeled) - np.mean(oob_scores_labeled)) / abs(np.mean(oob_scores_unlabeled))
        ) * 100
        if abs(absolute_percentage_difference) >= 5:
            print("percentage_difference", absolute_percentage_difference)

        # Perform the t-test (welsh)
        p_value = ttest_ind(oob_scores_labeled, oob_scores_unlabeled, alternative="less", equal_var=False).pvalue

        # Perform the mannwhitneyu test
        # p_value = mannwhitneyu(oob_scores_labeled, oob_scores_unlabeled, alternative="less").pvalue

        # Check if the result is statistically significant (alpha level = 0.05)
        if p_value <= 0.05:
            mean_oob_score_labeled = np.mean(oob_scores_labeled)
            mean_oob_score_unlabeled = np.mean(oob_scores_unlabeled)
            print(f"p_value {target_feature_name} {p_value} l: {mean_oob_score_labeled} ul: {mean_oob_score_unlabeled}")

            # Calculate the percentage difference between OOB scores
            absolute_percentage_difference = (
                abs((mean_oob_score_unlabeled - mean_oob_score_labeled) / abs(mean_oob_score_unlabeled)) * 100
            )
            print("absolute_percentage_difference", absolute_percentage_difference)

            # Calculate a metric based on the percentage difference and p-value
            print("metric", absolute_percentage_difference / log(p_value))
            print("---------")

    return mean_oob_score_labeled, mean_oob_score_unlabeled, p_value


# def calculate_oob_scores(target_feature_name, outer_cv_loop, meta_data):
#     oob_scores_labeled = []
#     oob_scores_unlabeled = []
#
#     # load preprocessed data
#     with open(f"data/{meta_data['data_name']}_preprocessed_cv_fold_outer{outer_cv_loop}_train.pkl",
#               "rb") as file:
#         train_df = pickle.load(file)
#
#     with open(f"data/{meta_data['data_name']}_preprocessed_cv_fold_outer{outer_cv_loop}_corr.pkl",
#               "rb") as file:
#         corr_matrix_df = pickle.load(file)
#
#     # prepare training data
#     y_train = ravel(train_df[target_feature_name])
#     # remove target feature from training data
#     x_train_df = train_df.loc[:, train_df.columns != target_feature_name]
#     # remove features correlated to target feature
#     x_train = preprocessing.remove_features_correlated_to_target_feature(x_train_df, corr_matrix_df, target_feature_name,
#                                                                          meta_data)
#
#     # inner cross-validation
#     for seed in meta_data['seed_list']:
#     # for inner_cv_loop in range(30):
#     # for inner_cv_loop in range(len(train_index)):
#         # # load preprocessed data
#         # with open(f"data/{meta_data['data_name']}_preprocessed_cv_fold_outer{outer_cv_loop}_inner{inner_cv_loop}_train.pkl", "rb") as file:
#         #     train_df = pickle.load(file)
#         #
#         # # with open(f"data/{meta_data['data_name']}_preprocessed_cv_fold_outer{outer_cv_loop}_inner{inner_cv_loop}_validation.pkl", "rb") as file:
#         # #     validation_df = pickle.load(file)
#         #
#         # with open(f"data/{meta_data['data_name']}_preprocessed_cv_fold_outer{outer_cv_loop}_inner{inner_cv_loop}_corr.pkl", "rb") as file:
#         #     corr_matrix_df = pickle.load(file)
#         #
#         # # data_split = train_df, validation_df, corr_matrix_df
#         # y_train = train_df[target_feature_name]
#         #
#         # # remove target feature from training data
#         # train_df = train_df.loc[:, train_df.columns != target_feature_name]
#         #
#         # # remove features correlated to target feature
#         # x_train = preprocessing.remove_correlated_features_to_target_feature(train_df, corr_matrix_df, target_feature_name, meta_data)
#
#     # inner cross-validation
#     # k_fold = LeaveOneOut()
#     # for inner_cv_loop, (inner_train_index, inner_test_index) in enumerate(k_fold.split(data_df.iloc[train_index, :])):
#     #     data_split = preprocessing.preprocess_data(
#     #         inner_train_index, inner_test_index, data_df, correlation_matrix=True
#     #     )
#         # start = datetime.utcnow()
#         # (
#         #     x_train,
#         #     y_train,
#         #     x_validation,
#         #     y_validation,
#         # ) = preprocessing.get_uncorrelated_train_and_validation_data(
#         #     data_split=data_split,
#         #     target_feature=target_feature_name,
#         #     labeled=labeled,
#         #     meta_data=meta_data,
#         # )
#         # x_train = train_df
#
#         # TODO move parameter setting to settings.toml
#         # # instantiate RandomForestRegressor with the best parameters
#         # clf = RandomForestRegressor()
#         # clf.set_params(**meta_data["selection_method"]["reverse_trees"]["best_params"])
#         clf1 = RandomForestRegressor(
#             warm_start=False,
#             max_features=None,
#             oob_score=mean_squared_error,
#             # criterion="absolute_error",
#             n_estimators=100,
#             random_state=seed,
#             min_samples_leaf=2,
#         )
#
#         # label is included in the training data
#         # if labeled:
#         assert "label" in x_train.columns
#         clf1.fit(x_train, ravel(y_train))
#         label_zero = clf1.feature_importances_[0] == 0
#         assert clf1.feature_importances_.size == x_train.shape[1]
#         oob_scores_labeled.append(clf1.oob_score_)
#         # print(target_feature_name, "pred", clf1.oob_prediction_)
#
#         # label was not included in the model
#         if label_zero:
#             return None, None
#
#         clf2 = clone(clf1)
#
#         # else:
#         # model without the label information in the training data
#         # remove target feature from training data
#         unlabeled_x_train = x_train.loc[:, x_train.columns != 'label']
#         assert "label" not in unlabeled_x_train.columns
#         clf2.fit(unlabeled_x_train, ravel(y_train))
#         oob_scores_unlabeled.append(clf2.oob_score_)
#         # print(target_feature_name, "pred", clf2.oob_prediction_)
#
#     return oob_scores_labeled, oob_scores_unlabeled


# def calculate_significance(target_feature_name, train_index, outer_cv_loop, meta_data):
#     score_labeled = 0
#     score_unlabeled = 0
#     p_value = None
#
#     # oob_scores_labeled = calculate_oob_scores(target_feature_name, train_index, outer_cv_loop, meta_data, clf, True)
#     # if oob_scores_labeled is not None:
#     #     oob_scores_unlabeled = calculate_oob_scores(target_feature_name, train_index, outer_cv_loop, meta_data, clf, False)
#     oob_scores_labeled, oob_scores_unlabeled = calculate_oob_scores(target_feature_name, train_index, outer_cv_loop,
#                                                                     meta_data)
#     if oob_scores_labeled is not None:
#         # oob_scores_unlabeled = calculate_oob_scores(target_feature_name, train_index, outer_cv_loop, meta_data, clf,
#
#         # Was training including the label better than without the label in the trainig data?
#         if abs(np.mean(oob_scores_labeled)) < abs(np.mean(oob_scores_unlabeled)):
#             # Perform the Wilcoxon signed-rank test
#             test_result = wilcoxon(oob_scores_labeled, oob_scores_unlabeled, alternative="less", method="exact",
#                                    correction=True)
#             p_value = test_result.pvalue
#
#             # Check if the result is statistically significant
#             if p_value <= 0.05:
#                 score_labeled = np.mean(oob_scores_labeled)
#                 score_unlabeled = np.mean(oob_scores_unlabeled)
#                 print(f"p_value {target_feature_name} {p_value} l: {score_labeled} ul: {score_unlabeled}")
#                 difference = np.mean(np.abs(oob_scores_unlabeled) - np.abs(oob_scores_labeled))
#                 print(difference, score_unlabeled - score_labeled)
#                 print("difference", difference)
#                 norm_diff = 1 - (score_labeled / score_unlabeled)
#                 print("norm_diff", norm_diff)
#
#                 # Calculate the percentage difference
#                 percentage_difference = ((score_labeled - score_unlabeled) / abs(score_unlabeled)) * 100
#                 print("percentage_difference", percentage_difference)
#                 metric = (difference / abs(score_labeled)) / p_value
#                 metric3 = norm_diff / p_value
#                 print("metric", metric, "metric3", metric3)
#                 print("---------")
#
#     return score_labeled, score_unlabeled, p_value


# def calculate_validation_metric(data_split, target_feature_name, meta_data, clf):
#     # initialize scores
#     score_unlabeled = 0
#     score_labeled = 0
#     p_value = None
#     (
#         x_train,
#         y_train,
#         x_validation,
#         y_validation,
#     ) = preprocessing.get_uncorrelated_train_and_validation_data(
#         data_split=data_split,
#         target_feature=target_feature_name,
#         labeled=True,
#         meta_data=meta_data,
#     )
#     # assert np.max(y_train) <= 1, str(np.max(y_train))
#
#     # predict the target feature including the label in the training data
#     clf.fit(x_train, ravel(y_train))
#     label_zero = clf.feature_importances_[0] == 0
#     assert clf.feature_importances_.size == x_train.shape[1]
#
#     # label was included in the model
#     if not label_zero:
#         score_labeled = clf.oob_score_
#         oob_scores_labeled = clf.oob_prediction_
#
#         # ensure at least one feature is uncorrelated to the target feature
#         assert x_train.shape[1] > 1
#
#         # predict the target feature without the label information
#         unlabeled_x_train = x_train.loc[:, x_train.columns != "label"]
#         assert "label" not in unlabeled_x_train.columns
#         assert unlabeled_x_train.shape[1] == x_train.shape[1] - 1
#         clf.fit(unlabeled_x_train, ravel(y_train))
#         score_unlabeled = clf.oob_score_
#         oob_scores_unlabeled = clf.oob_prediction_
#
#         # labeled training was not better than training without label
#         if (score_labeled >= score_unlabeled): # or (abs(score_unlabeled - score_labeled) < 0.1):
#             #  if score_labeled <= score_unlabeled or (
#             # # both scores are negative and don't have a significant distance
#             # score_unlabeled <= 0
#             # and score_labeled <= 0
#             # and abs(score_unlabeled) - abs(score_labeled) < 0.2
#             # ):
#             score_unlabeled = score_labeled = 0  # deselect feature
#         else:
#             test_result = mannwhitneyu(oob_scores_labeled, oob_scores_unlabeled)
#             p_value = test_result.pvalue
#             print("p_value m", p_value)
#             test_result = kruskal(oob_scores_labeled, oob_scores_unlabeled)
#             p_value = test_result.pvalue
#             print("p_value k", p_value)
#             test_result = ttest_ind(oob_scores_labeled, oob_scores_unlabeled)
#             p_value = test_result.pvalue
#             print("p_value t", p_value)
#
#     if score_labeled != 0:
#         print(
#             f"{target_feature_name} selected {abs(score_unlabeled - score_labeled)}, "
#             f"ul {score_unlabeled}, l {score_labeled}", {(abs(score_unlabeled - score_labeled)*100)/abs(score_unlabeled)}
#         )
#         print('-----------')
#     return score_labeled, score_unlabeled, p_value


def calculate_oob_errors_per_feature(data_df, meta_data, outer_cv_loop):
    scores_labeled_list = []
    scores_unlabeled_list = []
    p_values_list = []

    # # serial version
    # for target_feature_name in data_df.columns[1:]:
    #     score_labeled, score_unlabeled, p_value = calculate_mean_oob_scores_and_p_value(target_feature_name, outer_cv_loop, meta_data)
    #     scores_labeled_list.append(score_labeled)
    #     scores_unlabeled_list.append(score_unlabeled)
    #     p_values_list.append(p_value)

    # parallel version
    out = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=-1)(
        delayed(calculate_mean_oob_scores_and_p_value)(target_feature_name, outer_cv_loop, meta_data)
        for target_feature_name in data_df.columns[1:]
    )

    for score_labeled, score_unlabeled, p_value in out:
        scores_labeled_list.append(score_labeled)
        scores_unlabeled_list.append(score_unlabeled)
        p_values_list.append(p_value)

    assert len(scores_labeled_list) == len(scores_unlabeled_list) == data_df.shape[1] - 1  # exclude label column
    result_df = pd.DataFrame(data=scores_unlabeled_list, index=data_df.columns[1:], columns=["unlabeled"])
    result_df["labeled"] = scores_labeled_list
    result_df["p_values"] = p_values_list
    return result_df
