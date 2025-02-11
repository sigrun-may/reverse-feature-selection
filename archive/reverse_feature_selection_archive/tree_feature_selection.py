import math
import warnings

import cv_pruner
import lightgbm as lgb
import numpy as np
import optuna
from cv_pruner import Method
from optuna import TrialPruned, pruners
from optuna.samplers import TPESampler
from scipy.stats import trim_mean

import tree_model

warnings.filterwarnings("ignore")


def select_features(preprocessed_data_dict, outer_cv_loop, meta_data, extra_trees):
    """Optimize regularization parameter alpha for lasso regression."""

    def optuna_objective(trial):
        # if optuna_study_pruner.study_patience_pruner(
        #     trial, epsilon=0.001, warm_up_steps=20, patience=5
        # ) or optuna_study_pruner.study_no_improvement_pruner(
        #     trial,
        #     epsilon=0.01,
        #     warm_up_steps=30,
        #     number_of_similar_best_values=5,
        #     threshold=0.1,
        # ):
        #     print("study stopped")
        #     trial.study.stop()
        #     raise TrialPruned()

        # if "random" in target_feature_name or "pseudo" in target_feature_name:
        #     trial.study.stop()

        # prepare train/ test data
        # y_train = train_data_df["label"].values.reshape(-1, 1)
        # x_train = train_data_df.drop(columns="label")
        # x_test = test_data_df.drop(columns="label").values
        # y_test = test_data_df["label"].values
        #
        # train_data = lgb.Dataset(x_train, label=y_train)
        # test_data = lgb.Dataset(x_test, label=y_test)

        # x_train = train_data_df.drop(columns="label")
        # x_test = test_data_df.drop(columns="label").values
        # train_data = lgb.Dataset(x_train, label=train_data_df["label"])
        # test_data = lgb.Dataset(x_test, label=test_data_df["label"])

        _, train_data_df, _ = preprocessed_data_dict["preprocessed_data_list"][0]
        parameters = dict(
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, math.floor(train_data_df.shape[0] / 2)),
            lambda_l1=trial.suggest_uniform("lambda_l1", 0.0, 3),
            min_gain_to_split=trial.suggest_uniform("min_gain_to_split", 0, 5),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            bagging_fraction=trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            extra_trees=extra_trees,
            objective="binary",
            metric="binary_logloss",
            boosting_type="rf",
            verbose=0,
        )
        # num_leaves must be smaller than 2^max_depth
        # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree
        max_num_leaves = 2 ** parameters["max_depth"] - 1
        max_num_leaves = min(max_num_leaves, 90)
        parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)

        validation_metric_history = []
        feature_importances_list = []
        shap_values_list = []

        # cross validation for the optimization of alpha
        for fold_index, (test_data_df, train_data_df, _) in enumerate(preprocessed_data_dict["preprocessed_data_list"]):
            model, prune = tree_model.train_model(parameters, test_data_df, train_data_df, "label", _)
            # model = lgb.train(
            #     parameters,
            #     train_data,
            #     valid_sets=[test_data],
            #     callbacks=[lgb.record_evaluation({})],  # stop verbose
            # )

            if prune:
                raise TrialPruned()

            validation_metric_history.append(model.best_score["valid_0"]["binary_logloss"])

            if cv_pruner.should_prune_against_threshold(
                current_step_of_complete_nested_cross_validation=fold_index,
                folds_outer_cv=meta_data["cv"]["n_outer_folds"],
                folds_inner_cv=meta_data["cv"]["n_inner_folds"],
                validation_metric_history=validation_metric_history,
                threshold_for_pruning=meta_data["selection_method"]["trees"]["pruner_threshold"],
                direction_to_optimize_is_minimize=True,
                optimal_metric=0,
                method=Method.OPTIMAL_METRIC,
            ):
                return trim_mean(validation_metric_history, proportiontocut=0.2)

            feature_importances_list.append(model.feature_importance(importance_type="gain"))

            # calculate shap values

            # explainer = shap.TreeExplainer(model)
            # shap_values = explainer(x_test)
            # raw_shap_values = np.abs(shap_values.values)[:, :, 0]
            # shap_values_list.append(np.sum(raw_shap_values, axis=0))  # TODO SHAP mean or sum
            x_test = test_data_df.drop(columns="label").values
            shap_values_list.append(tree_model.get_shap_list(model, x_test))

        trial.set_user_attr("shap_values", np.array(shap_values_list))
        trial.set_user_attr("macro_feature_importances", np.array(feature_importances_list))

        #     if (
        #         meta_data_dict["selection_method"]["rf"]["shap_test"] is None
        #         and meta_data_dict["selection_method"]["rf"]["shap_train"] is None
        #     ):
        #         cumulated_importances = model.feature_importance(importance_type="gain")
        #     else:
        #         explainer = shap.TreeExplainer(model)
        #         if meta_data_dict["selection_method"]["rf"]["shap_test"]:
        #             shap_values = explainer(x_test)
        #
        #         elif meta_data_dict["selection_method"]["rf"]["shap_train"]:
        #             shap_values = explainer(x_train)
        #         raw_shap_values = np.abs(shap_values.values)[:, :, 0]
        #         cumulated_importances = np.sum(raw_shap_values, axis=0)
        #
        #     summed_importances.append(cumulated_importances)
        #
        #     # monitor robustness of selection
        #     used_features = np.zeros_like(cumulated_importances)
        #     used_features[cumulated_importances.nonzero()] = 1
        #     used_features_list.append(used_features)
        #
        # feature_idx = np.sum(np.array(summed_importances), axis=0)
        # unlabeled_feature_names = feature_names[1:]
        # selected_features = unlabeled_feature_names[feature_idx.nonzero()]
        # nonzero_shap_values = feature_idx[feature_idx.nonzero()]
        # assert len(selected_features) == feature_idx.nonzero()[0].size
        #
        # robustness_array = np.sum(np.array(used_features_list), axis=0)
        # feature_robustness = robustness_array[robustness_array.nonzero()]
        # assert (
        #     feature_robustness.shape
        #     == nonzero_shap_values.shape
        #     == selected_features.shape
        # )
        # trial.set_user_attr("shap_values", nonzero_shap_values)
        # trial.set_user_attr("selected_features", selected_features)
        # trial.set_user_attr("robustness_selected_features", feature_robustness)
        # trial.set_user_attr("robustness_all_features", robustness_array)
        # trial.set_user_attr("feature_importances", np.array(used_features_list))

        return trim_mean(validation_metric_history, proportiontocut=0.2)

    # try:
    #     optuna.study.delete_study(f"{target_feature_name}_iteration_{test_indices}", storage = "sqlite:///optuna_db.db")
    # except:
    #     print("new study")

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"standard_rf_iteration_{outer_cv_loop}",
        direction="minimize",
        sampler=TPESampler(
            multivariate=True,
        ),
        pruner=pruners.SuccessiveHalvingPruner(
            min_resource="auto",
            reduction_factor=3,
            min_early_stopping_rate=2,
            bootstrap_count=0,
        ),
    )
    if meta_data["parallel"]["cluster"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.select_feature_subset(
        optuna_objective,
        n_trials=meta_data["selection_method"]["trees"]["trials"],
    )

    micro_feature_importance = lgb.train(
        study.best_params,
        lgb.Dataset(
            data=preprocessed_data_dict["micro_train"].iloc[:, 1:],
            label=preprocessed_data_dict["micro_train"].iloc[:, 0],
        ),
        callbacks=[lgb.record_evaluation({})],  # stop verbose
    ).feature_importance(importance_type="gain")

    # check if study.best_value is available and at least one trial was completed
    try:
        best = study.best_value
    except:
        return {"NONE": 0}

    return {
        "shap_values": study.best_trial.user_attrs["shap_values"],
        "macro_feature_importances": study.best_trial.user_attrs["macro_feature_importances"],
        "micro_feature_importance": micro_feature_importance,
    }

    # try:
    #     return (
    #         dict(
    #             zip(
    #                 study.best_trial.user_attrs["selected_features"],
    #                 zip(
    #                     study.best_trial.user_attrs["shap_values"],
    #                     study.best_trial.user_attrs["robustness_selected_features"],
    #                 ),
    #             )
    #         ),
    #         study.best_trial.user_attrs["robustness_all_features"],
    #         study.best_trial.user_attrs["feature_importances"],
    #     )
    # except:
    #     return {"NONE": 0}
