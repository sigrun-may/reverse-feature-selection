import math
from optuna.samplers import TPESampler
from optuna import TrialPruned, pruners
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy.stats import trim_mean
import cv_pruner
from cv_pruner import Method

import settings


def select_features(
    transformed_test_train_splits_dict,
    outer_cv_loop
):
    """Optimize regularization parameter alpha for lasso regression."""

    def optuna_objective(trial):

        # if "random" in target_feature_name or "pseudo" in target_feature_name:
        #     trial.study.stop()

        if trial.number >= 10:
            try:
                trial.study.best_value
            except:
                # print('no results for more than 10 trials')
                # print(target_feature_name)
                trial.study.stop()

        validation_metric_history = []
        all_shap_values = []
        sum_of_all_shap_values = []

        feature_names = transformed_test_train_splits_dict["feature_names"].values
        transformed_data = transformed_test_train_splits_dict["transformed_data"]

        # cross validation for the optimization of alpha
        for fold_index, (test, train, train_correlation_matrix_complete) in enumerate(
            transformed_data
        ):
            # TODO pandas notwendig?
            train_data_df = pd.DataFrame(train, columns=feature_names)
            test_data_df = pd.DataFrame(test, columns=feature_names)

            # prepare train/ test data
            y_train = train_data_df["label"].values.reshape(-1, 1)
            x_train = train_data_df.drop(columns="label")
            x_test = test_data_df.drop(columns="label").values
            y_test = test_data_df["label"].values

            train_data = lgb.Dataset(x_train, label=y_train)
            test_data = lgb.Dataset(x_test, label=y_test)

            # parameters for model training to combat overfitting
            parameters = dict(
                min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 2, math.floor(x_train.shape[0] / 2)),
                lambda_l1 = trial.suggest_uniform("lambda_l1", 0.0, 3),
                min_gain_to_split = trial.suggest_uniform("min_gain_to_split", 0, 5),
                max_depth = trial.suggest_int("max_depth", 2, 20),
                bagging_fraction = trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
                bagging_freq = trial.suggest_int("bagging_freq", 1, 10),
                extra_trees = True,
                objective = "binary",
                metric = "binary_logloss",
                boosting_type = "rf",
                verbose = -1,
            )

            # num_leaves must be greater than 2^max_depth
            max_num_leaves = 2 ** parameters["max_depth"] - 1
            if max_num_leaves < 90:
                parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)
            else:
                parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, 90)

            model = lgb.train(
                parameters,
                train_data,
                valid_sets = [test_data],
                verbose_eval = False,
            )

            if cv_pruner.check_no_features_selected(model.feature_importance(importance_type = 'gain')):
                raise TrialPruned()

            validation_metric_history.append(model.best_score["valid_0"]["binary_logloss"])

            if cv_pruner.check_against_threshold(
                current_step_of_complete_nested_cross_validation = fold_index,
                folds_outer_cv= train_data_df.shape[0] + test_data_df.shape[0],  # loo
                folds_inner_cv= settings.N_FOLDS_INNER_CV,
                validation_metric_history= validation_metric_history,
                threshold_for_pruning= 0.3,
                direction_to_optimize_is_minimize= True,
                optimal_metric= 0,
                method= Method.OPTIMAL_METRIC,
            ):
                return trim_mean(validation_metric_history, proportiontocut = 0.2)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(x_train)
            raw_shap_values = np.abs(shap_values.values)[:, :, 0]
            # cumulated_shap_values = np.stack(raw_shap_values, axis=0)
            # all_shap_values.append(cumulated_shap_values)
            sum_of_all_shap_values.append(np.sum(raw_shap_values, axis=0))

        feature_idx = np.sum(np.array(sum_of_all_shap_values), axis=0)
        unlabeled_feature_names = feature_names[1:]
        selected_features = unlabeled_feature_names[feature_idx.nonzero()]
        nonzero_shap_values = feature_idx[feature_idx.nonzero()]
        assert len(selected_features) == feature_idx.nonzero()[0].size

        trial.set_user_attr("shap_values", nonzero_shap_values)
        trial.set_user_attr("selected_features", selected_features)
        return trim_mean(validation_metric_history, proportiontocut = 0.2)

    # try:
    #     optuna.study.delete_study(f"{target_feature_name}_iteration_{outer_cv_loop}", storage = "sqlite:///optuna_db.db")
    # except:
    #     print("new study")

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"standard_rf_iteration_{outer_cv_loop}",
        direction="minimize",
        sampler=TPESampler(
            multivariate=True,
            n_startup_trials=3,
            consider_magic_clip=True,
            constant_liar=True,
            warn_independent_sampling = False,
        ),
        pruner=pruners.SuccessiveHalvingPruner(
            min_resource="auto",
            reduction_factor=3,
            min_early_stopping_rate=2,
            bootstrap_count=0,
        ),
    )
    # optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        # n_trials = 40,
        n_trials=15,
        n_jobs=1,
    )
    relevant_features_dict = dict(
        label=(0, 0, study.best_trial.user_attrs["shap_values"])
    )
    intermediate_result = dict(relevant_features_dict=relevant_features_dict)
    intermediate_result["test_train_indices"] = {}
    # return study.best_trial.user_attrs["label_coefficients"]
    return dict(
        zip(
            study.best_trial.user_attrs["selected_features"],
            study.best_trial.user_attrs["shap_values"],
        )
    )

