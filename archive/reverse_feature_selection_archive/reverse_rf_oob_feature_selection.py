import joblib
import numpy as np
import optuna
from optuna import TrialPruned
from sklearn.ensemble import RandomForestRegressor

#
import optuna_study_pruner
from utils import sort_list_of_tuples_by_index


def calculate_oob(
    parameters,
    target_feature_name,
    deselected_features,
    complete_train_data_df,
    include_label,
    trial,
    meta_data,
):
    original_feature_names = complete_train_data_df.columns
    feature_names = []
    for feature_name in original_feature_names:
        if "cluster" in feature_name:
            feature_names.append(feature_name.replace("cluster_", ""))
        else:
            feature_names.append(feature_name)

    assert feature_names[0] == "label"

    try:
        train_correlation_matrix_complete = joblib.load(meta_data["clustered_correlation_matrix_path"])
    except:
        raise ValueError("no correlation matrix")

    assert train_correlation_matrix_complete.shape[0] > 0
    assert train_correlation_matrix_complete.shape[1] > 0
    assert set(train_correlation_matrix_complete.columns) == set(feature_names[1:])

    # prepare train/ test data
    train_data_df = complete_train_data_df.copy()
    y_train = complete_train_data_df[target_feature_name].values.reshape(-1, 1)

    # # remove irrelevant features from train_correlation_matrix
    # if meta_data["selection_method"]["reverse_lasso"]["remove_deselected"]:
    #     train_correlation_matrix = train_correlation_matrix_complete.drop(
    #         labels=deselected_features,
    #         axis=0,
    #         inplace=False,
    #     )
    #     train_correlation_matrix.drop(
    #         labels=deselected_features,
    #         axis=1,
    #         inplace=True,
    #     )
    #     assert (
    #         train_correlation_matrix_complete.shape[1] - len(deselected_features)
    #         == train_correlation_matrix.shape[1]
    #     )
    #     assert (
    #         train_correlation_matrix_complete.shape[0] - len(deselected_features)
    #         == train_correlation_matrix.shape[0]
    #     )
    #
    #     # remove irrelevant features from test and train data
    #     train_data_df.drop(columns=deselected_features, inplace=True)
    #     assert train_data_df.shape[1] == train_data_df.shape[1] - len(deselected_features)

    # How Correlations Influence Lasso Prediction, April 2012IEEE Transactions on Information Theory 59(3)
    # DOI: 10.1109/TIT.2012.2227680
    # find features correlated to the target_feature from test/ train data
    correlated_features = [
        (feature, correlation_coefficient)
        for feature, correlation_coefficient in train_correlation_matrix_complete[target_feature_name].items()
        if abs(correlation_coefficient) > meta_data["selection_method"]["reverse_lasso"]["correlation_threshold"]
    ]

    correlated_feature_names = list(map(list, zip(*correlated_features)))[0]

    # check if train would keep at least one feature after removing label and target_feature
    if train_data_df.shape[1] - len(correlated_features) < 3:
        absolute_correlations = [
            (feature_name, abs(correlation_coefficient))
            for feature_name, correlation_coefficient in correlated_features
        ]
        # keep the feature with the lowest correlation to the target feature
        sorted_correlated_features = sort_list_of_tuples_by_index(absolute_correlations, index=1, ascending=True)
        min_correlated_feature = sorted_correlated_features[0][0]
        correlated_feature_names.remove(min_correlated_feature)
        assert len(absolute_correlations) - 1 == len(correlated_feature_names)

    # if len(correlated_features) > 0:
    #     if target_feature_name in correlated_feature_names:
    # correlated_feature_names.remove(target_feature_name)

    if not include_label:
        # append label to the list of features to remove
        correlated_feature_names.append("label")

    # remove features correlated to the target_feature from test/ train data and the label if it is not included
    train_data_df.drop(columns=correlated_feature_names, inplace=True)
    # assert train_data_df.shape[1] == train.shape[1] - len(
    #     deselected_features
    # ) - len(correlated_features)
    #
    # assert test_data_df.shape[1] == test.shape[1] - len(deselected_features) - len(
    #     correlated_features
    # )
    if not train_data_df.shape[1] == complete_train_data_df.shape[1] - len(correlated_feature_names):
        print("d")
    assert train_data_df.shape[1] == complete_train_data_df.shape[1] - len(correlated_feature_names), (
        str(len(correlated_feature_names)) + "  train_data_df.shape[1] " + str(train_data_df.shape[1])
    )

    # # prepare train/ test data
    # y_train = train_data_df[target_feature_name].values.reshape(-1, 1)
    # x_train = train_data_df.drop(columns=target_feature_name)

    assert train_data_df.shape[1] >= 1

    # build model
    model = RandomForestRegressor(
        max_depth=4,
        # criterion="absolute_error",
        min_impurity_decrease=parameters["min_impurity_decrease"],
        oob_score=True,
    )
    model.fit(train_data_df, np.ravel(y_train))

    # save number of nonzero coefficients to calculate r2 adjusted
    assert len(model.feature_importances_) == train_data_df.shape[1]

    if include_label:
        # prune trials if label coefficient is zero
        label_coefficient = model.feature_importances_[0]
        if label_coefficient == 0:
            raise TrialPruned()

    return model.oob_score_


def optimize(
    train_data,
    target_feature_name,
    deselected_features,
    outer_cv_loop,
    meta_data,
):
    """Optimize regularization parameter alpha for lasso regression."""

    def optuna_objective(trial):
        # optuna_study_pruner.study_no_trial_completed_pruner(trial, warm_up_steps=10)
        # optuna_study_pruner.study_no_improvement_pruner(
        #     trial,
        #     epsilon=0.001,
        #     warm_up_steps=10,
        #     number_of_similar_best_values=5,
        #     threshold=0.1,
        # )

        optuna_study_pruner.insufficient_results_study_pruner(trial, warm_up_steps=15, threshold=0.05)
        # if 'random' in target_feature_name:
        #     return 0

        parameters = dict(min_impurity_decrease=trial.suggest_uniform("min_impurity_decrease", 0.0, 3))

        return calculate_oob(
            parameters=parameters,
            target_feature_name=target_feature_name,
            deselected_features=deselected_features,
            complete_train_data_df=train_data,
            include_label=True,
            trial=trial,
            meta_data=meta_data,
        )

    # try:
    #     optuna.study.delete_study(f"{target_feature_name}_iteration_{test_indices}", storage = "sqlite:///optuna_db.db")
    # except:
    #     print("new study")

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"{target_feature_name}_iteration_{outer_cv_loop}",
        direction="maximize",
    )
    if meta_data["parallel"]["cluster"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        # n_trials = 40,
        n_trials=meta_data["selection_method"]["reverse_lasso"]["trials"],
    )
    # # if study.best_value > 0:
    # #     fig = optuna.visualization.plot_optimization_history(study)
    # #     fig.show()
    #

    # check if study.best_value is available and at least one trial was completed
    try:
        if study.best_value <= 0:  # Was r2 adjusted greater than zero?
            return 0, 0
    except:
        return 0, 0

    # calculate r2 without the label in the training data for the same alpha
    oob_score = calculate_oob(
        study.best_params,
        target_feature_name,
        deselected_features,
        train_data,
        include_label=False,
        trial=None,
        meta_data=meta_data,
    )
    print(
        target_feature_name,
        study.best_value,
        oob_score,
        # study.best_trial.user_attrs["r2_score"],
        # study.best_trial.user_attrs["median_number_of_features_in_model"],
    )

    # check if study.best_value is available and at least one trial was completed
    try:
        if study.best_value <= 0:  # Was r2 adjusted greater than zero?
            return 0, 0
    except:
        return 0, 0

    return (
        study.best_value,
        oob_score,
    )


def select_features(train_data, outer_cv_loop_iteration, meta_data):
    # calculate relevance for each feature
    deselected_features_list = []
    robustness_vector = []
    selected_features_dict = {}
    for target_feature_name in train_data.columns:
        # exclude the label
        if target_feature_name == "label":
            continue

        # get performance of target feature
        # TODO label_coefficients_list needed?
        oob_labeled, oob_unlabeled = optimize(
            train_data,
            target_feature_name,
            deselected_features_list,
            outer_cv_loop=outer_cv_loop_iteration,
            meta_data=meta_data,
        )

        # save switched to maximized metric
        selected_features_dict[target_feature_name] = (
            oob_unlabeled,
            oob_labeled,
        )

    assert len(selected_features_dict) == len(train_data.columns) - 1  # exclude the label
    return selected_features_dict  # , np.array(robustness_vector)
