import numpy as np
import pandas as pd
import optuna
from optuna import TrialPruned
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score
import celer

#
import optuna_study_pruner
from utils import sort_list_of_tuples_by_index


def calculate_adjusted_r2(y, predicted_y, number_of_coefficients):
    # calculate the coefficient of determination, r2:
    # The proportion of the variance in the dependent variable that is predictable from the independent variable(s).
    # It provides selected_feature_subset_list measure of how well observed outcomes are replicated by the model,
    # based on the proportion of total variation of outcomes explained by the model.
    r2 = r2_score(y, predicted_y)
    sample_size = len(y)

    # adjusted_r2 = 1 - (1 - r2) * (sample_size - 1) / (sample_size - (np.max(number_of_coefficients)) - 1)
    adjusted_r2 = 1 - (
        ((1 - r2) * (sample_size - 1))
        / (sample_size - (np.median(number_of_coefficients)) - 1)
    )

    # Adj r2 = 1-(1-R2)*(n-1)/(n-p-1)
    # assume n = number of sample size , p = number of independent variables
    # The higher the corrected R², the better the model fits the data.
    return adjusted_r2


def calculate_r2(
    alpha,
    target_feature_name,
    deselected_features,
    data_dict,
    include_label,
    trial,
    meta_data,
):
    predicted_y = []
    true_y = []
    number_of_coefficients_list = []
    label_coefficients = []
    r2_list = []

    feature_names = data_dict["feature_names"].values
    assert feature_names[0] == "label"
    transformed_data = data_dict["transformed_data"]

    # cross validation for the optimization of alpha
    for test, train, train_correlation_matrix_complete in transformed_data:

        train_correlation_matrix = train_correlation_matrix_complete
        train_data_df = pd.DataFrame(train, columns=feature_names)
        test_data_df = pd.DataFrame(test, columns=feature_names)

        # remove irrelevant features from train_correlation_matrix
        if meta_data["selection_method"]["reverse_lasso"]["remove_deselected"]:
            train_correlation_matrix = train_correlation_matrix_complete.drop(
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
                train_correlation_matrix_complete.shape[1] - len(deselected_features)
                == train_correlation_matrix.shape[1]
            )
            assert (
                train_correlation_matrix_complete.shape[0] - len(deselected_features)
                == train_correlation_matrix.shape[0]
            )

            # remove irrelevant features from test and train data
            train_data_df.drop(columns=deselected_features, inplace=True)
            assert train_data_df.shape[1] == train.shape[1] - len(deselected_features)

            test_data_df.drop(columns=deselected_features, inplace=True)
            assert test_data_df.shape[1] == test.shape[1] - len(deselected_features)

        # How Correlations Influence Lasso Prediction, April 2012IEEE Transactions on Information Theory 59(3)
        # DOI: 10.1109/TIT.2012.2227680
        # find features correlated to the target_feature from test/ train data
        correlated_features = [
            (feature, correlation_coefficient)
            for feature, correlation_coefficient in train_correlation_matrix[
                target_feature_name
            ].items()
            if abs(correlation_coefficient)
            > meta_data["selection_method"]["reverse_lasso"]["correlation_threshold"]
        ]

        # check if train would keep at least one feature after removing label and target_feature
        if train_data_df.shape[1] - len(correlated_features) < 3:
            # keep the feature with the lowest correlation to the target feature
            sorted_correlated_features = sort_list_of_tuples_by_index(
                correlated_features, index=1, ascending=True
            )
            min_correlated_feature = sorted_correlated_features[0][0]
            correlated_features.remove(min_correlated_feature)

        correlated_feature_names = list(map(list, zip(*correlated_features)))[0]

        # if len(correlated_features) > 0:
        #     if target_feature_name in correlated_feature_names:
        correlated_feature_names.remove(target_feature_name)

        if not include_label:
            # append label to the list of features to remove
            correlated_feature_names.append("label")

        # remove features correlated to the target_feature from test/ train data and the label if it is not included
        train_data_df.drop(columns=correlated_feature_names, inplace=True)
        test_data_df.drop(columns=correlated_feature_names, inplace=True)
        # assert train_data_df.shape[1] == train.shape[1] - len(
        #     deselected_features
        # ) - len(correlated_features)
        #
        # assert test_data_df.shape[1] == test.shape[1] - len(deselected_features) - len(
        #     correlated_features
        # )

        assert train_data_df.shape[1] == train.shape[1] - len(correlated_feature_names)
        assert test_data_df.shape[1] == test.shape[1] - len(correlated_feature_names)

        # prepare train/ test data
        y_train = train_data_df[target_feature_name].values.reshape(-1, 1)
        x_train = train_data_df.drop(columns=target_feature_name)
        x_test = test_data_df.drop(columns=target_feature_name).values
        y_test = test_data_df[target_feature_name].values
        true_y.extend(test_data_df[target_feature_name].values)

        assert x_train.shape[1] >= 1
        # build LASSO model
        lasso = celer.Lasso(alpha=alpha, verbose=0)
        lasso.fit(x_train.values, y_train)

        # sklearn lasso
        # lasso = Lasso(alpha = alpha, fit_intercept = True, positive = False)
        # lasso.fit(np.asfortranarray(x_train), y_train)

        # save number of non zero coefficients to calculate r2 adjusted
        assert len(lasso.coef_) == x_train.shape[1]
        number_of_coefficients = sum(lasso.coef_ != 0.0)
        # TODO still needed?
        number_of_coefficients_list.append(number_of_coefficients)

        if include_label:
            # prune trials if label coefficient is zero
            label_coefficient = lasso.coef_[0]
            if label_coefficient == 0:
                raise TrialPruned()
            # TODO still needed?
            label_coefficients.append(label_coefficient)

        # predict y_test
        predicted_y.extend(lasso.predict(x_test))
        r2_list.append(r2_score(y_test, lasso.predict(x_test)))

    # assume n = number of samples , p = number of independent variables
    # adjusted_r2 = 1-(1-R2)*(n-1)/(n-p-1)
    sample_size = len(true_y)
    adjusted_r2 = 1 - (
        ((1 - r2_score(true_y, predicted_y)) * (sample_size - 1))
        / (sample_size - np.median(number_of_coefficients_list) - 1)
    )
    if include_label:  # TODO prüfen, ob das noch gebraucht wird
        trial.set_user_attr("label_coefficients", label_coefficients)
        # trial.set_user_attr("r2_score", r2_score(true_y, predicted_y))
        trial.set_user_attr(
            "median_number_of_features_in_model", np.median(number_of_coefficients_list)
        )
    # return adjusted_r2
    # return r2_score(true_y, predicted_y)
    return np.mean(r2_list)


def optimize(
    transformed_test_train_splits_dict,
    target_feature_name,
    deselected_features,
    outer_cv_loop,
    meta_data,
):
    """Optimize regularization parameter alpha for lasso regression."""

    def optuna_objective(trial):
        optuna_study_pruner.study_no_trial_completed_pruner(trial, warm_up_steps=10)
        optuna_study_pruner.study_no_improvement_pruner(
            trial,
            epsilon=0.001,
            warm_up_steps=10,
            number_of_similar_best_values=5,
            threshold=0.1,
        )

        optuna_study_pruner.insufficient_results_study_pruner(
            trial, warm_up_steps=10, threshold=0.05
        )

        return calculate_r2(
            # alpha=trial.suggest_discrete_uniform("alpha", 0.001, 1.0, 0.001),
            alpha=trial.suggest_uniform("alpha", 0.01, 1.0),
            target_feature_name=target_feature_name,
            deselected_features=deselected_features,
            data_dict=transformed_test_train_splits_dict,
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
        # The higher R², the better the model fits the data.
        sampler=TPESampler(
            n_startup_trials=3,
        ),
    )

    reverse_lasso = (
        dict(
            trials=20,
            pruner_patience=None,
            pruner_threshold=0.1,
            threshold_correlations=0.2,
            remove_deselected=False,
        ),
    )
    if meta_data["parallel"]["cluster"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        # n_trials = 40,
        n_trials=meta_data["selection_method"]["reverse_lasso"]["trials"],
    )
    # if study.best_value > 0:
    #     fig = optuna.visualization.plot_optimization_history(study)
    #     fig.show()

    # check if study.best_value is available and at least one trial was completed
    try:
        if study.best_value <= 0:  # Was r2 adjusted greater than zero?
            return 0, 0, None
    except:
        return 0, 0, None

    # calculate r2 without the label in the training data for the same alpha
    r2 = calculate_r2(
        study.best_params["alpha"],
        target_feature_name,
        deselected_features,
        transformed_test_train_splits_dict,
        include_label=False,
        trial=None,
        meta_data=meta_data,
    )
    print(
        target_feature_name,
        study.best_value,
        r2,
        # study.best_trial.user_attrs["r2_score"],
        study.best_trial.user_attrs["median_number_of_features_in_model"],
    )
    return (
        r2,
        study.best_value,
        study.best_trial.user_attrs["label_coefficients"],
    )


def select_features(
    transformed_test_train_splits_dict, outer_cv_loop_iteration, meta_data
):
    # calculate relevance for each feature
    deselected_features_list = []
    robustness_vector = []
    selected_features_dict = {}
    for target_feature_name in transformed_test_train_splits_dict["feature_names"]:

        # exclude the label
        if target_feature_name == "label":
            continue

        # get performance of target feature
        # TODO label_coefficients_list needed?
        (r2_adjusted_unlabeled, r2_adjusted, label_coefficients_list,) = optimize(
            transformed_test_train_splits_dict,
            target_feature_name,
            deselected_features_list,
            outer_cv_loop=outer_cv_loop_iteration,
            meta_data=meta_data,
        )

        if r2_adjusted_unlabeled < r2_adjusted:
            # select feature
            selected_features_dict[target_feature_name] = (
                r2_adjusted_unlabeled,
                r2_adjusted,
            )
            robustness_vector.append(1)
            # selected_features_dict[target_feature_name] = (
            #     r2_adjusted - r2_adjusted_unlabeled
            # )
        else:
            # exclude irrelevant feature from training data
            deselected_features_list.append(target_feature_name)
            robustness_vector.append(0)

    assert (
        len(robustness_vector)
        == len(transformed_test_train_splits_dict["feature_names"]) - 1
    )  # exclude the label
    return selected_features_dict, np.array(robustness_vector)
