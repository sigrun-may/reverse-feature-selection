import cv_pruner
import numpy as np
import optuna
import pandas as pd
import shap
from optuna import TrialPruned
from optuna.samplers import TPESampler
from relaxed_lasso import RelaxedLasso
from sklearn.metrics import r2_score


def select_features(
    preprocessed_data_dict,
    outer_cv_loop: int,
    meta_data,
    # preprocessed_data_dict: Dict[str, List[Union[Tuple[Any], str]]],
):
    """Select feature subset for best value of regularization parameter alpha.

    Args:
        preprocessed_data_dict: yj +pearson train data
        outer_cv_loop: test index of outer cross-validation loop

    Returns: selected features + weights
    """

    def optuna_objective(trial):
        """Optimize regularization parameter alpha for lasso regression."""

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

        predicted_y = []
        true_y = []
        coefficients_list = []
        shap_values_list = []
        alpha = trial.suggest_uniform("alpha", 0.01, 1.0)
        theta = trial.suggest_uniform("theta", 0.01, 1.0)

        feature_names = preprocessed_data_dict["feature_names"]
        transformed_data = preprocessed_data_dict["transformed_data"]

        # cross validation for the optimization of alpha
        for test, train, _ in transformed_data:
            train_data_df = pd.DataFrame(train, columns=feature_names)
            test_data_df = pd.DataFrame(test, columns=feature_names)

            # prepare train/ test data
            y_train = train_data_df["label"].values.reshape(-1, 1)
            x_train = train_data_df.drop(columns="label")
            x_test = test_data_df.drop(columns="label").values
            y_test = test_data_df["label"].values
            true_y.extend(test_data_df["label"].values)

            # build LASSO model
            lasso = RelaxedLasso(
                alpha=alpha,
                theta=theta,
                selection="random",
                verbose=-1,
                # alpha=trial.suggest_discrete_uniform("alpha", 0.001, 1.0,
                # 0.001),
            )
            lasso.fit(x_train.values, y_train)

            # sklearn lasso
            # lasso = Lasso(alpha = trial.suggest_uniform("alpha", 0.01, 1.0), fit_intercept = True, positive = False)
            # lasso.fit(np.asfortranarray(x_train), y_train)
            if cv_pruner.no_features_selected(lasso.coef_):
                raise TrialPruned()

            coefficients_list.append(np.abs(lasso.coef_))

            predicted_y_test = lasso.predict(x_test)
            predicted_y.extend(predicted_y_test)

            explainer = shap.explainers.Linear(lasso, x_train)
            shap_values = explainer(x_test)
            mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
            shap_values_list.append(mean_shap_values)

        assert len(shap_values_list) == len(coefficients_list)
        trial.set_user_attr("shap_values", np.array(shap_values_list))
        trial.set_user_attr("macro_feature_importances", np.array(coefficients_list))

        #     if (
        #         meta_data["selection_method"]["lasso"]["shap_test"] is None
        #         and meta_data["selection_method"]["lasso"]["shap_train"] is None
        #     ):
        #         cumulated_importances = lasso.coef_
        #     else:
        #         explainer = shap.explainers.Linear(lasso, x_train)
        #         if meta_data["selection_method"]["lasso"]["shap_test"]:
        #             shap_values = explainer(x_test)
        #
        #         elif meta_data["selection_method"]["lasso"]["shap_train"]:
        #             shap_values = explainer(x_train)
        #         cumulated_importances = np.sum(np.abs(shap_values.values), axis=0)
        #
        #     summed_importances.append(cumulated_importances)
        #     # TODO unterschied coeff zu stacked SHAP
        #
        #     # monitor robustness of selection
        #     used_features = np.zeros_like(cumulated_importances)
        #     used_features[cumulated_importances.nonzero()] = 1
        #     used_features_list.append(used_features)
        #
        #     # predict y_test
        #     # predicted_y_test =
        #     # predicted_y.append(predicted_y_test[0])
        #     r2_list.append(r2_score(y_test, lasso.predict(x_test)))
        #     # predicted_y.extend(predicted_y_test)
        #
        # feature_idx = np.sum(np.array(summed_importances), axis=0)
        # unlabeled_feature_names = feature_names.drop(["label"])
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
        #
        # trial.set_user_attr("shap_values", nonzero_shap_values)
        # trial.set_user_attr("selected_features", selected_features)
        # trial.set_user_attr("robustness_selected_features", feature_robustness)
        # trial.set_user_attr("robustness_all_features", robustness_array)
        # trial.set_user_attr("feature_importances", np.array(used_features_list))
        # trial.set_user_attr("shap_values_train", np.array(used_features_list))
        # trial.set_user_attr("shap_values_test", np.array(used_features_list))
        # # r2 = r2_score(true_y, predicted_y_proba)
        #
        # # assume n = number of samples , p = number of independent variables
        # # adjusted_r2 = 1-(1-R2)*(n-1)/(n-p-1)
        # sample_size = len(true_y)
        # adjusted_r2 = 1 - (
        #     ((1 - r2_score(true_y, predicted_y)) * (sample_size - 1))
        #     / (sample_size - (np.median(number_of_coefficients)) - 1)
        # )
        return r2_score(true_y, predicted_y)
        # return r2_score(true_y, predicted_y)
        # return calculate_adjusted_r2(true_y, predicted_y, number_of_coefficients)

    # try:
    #     optuna.study.delete_study(f"{target_feature_name}_iteration_{test_indices}", storage = "sqlite:///optuna_db.db")
    # except:
    #     print("new study")

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"standard_lasso_iteration_{outer_cv_loop}",
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=3,
        ),
    )
    if meta_data["parallel"]["cluster"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.optimize(
        optuna_objective,
        n_trials=meta_data["selection_method"]["lasso"]["trials"],
    )
    # build LASSO model
    lasso_all = RelaxedLasso(
        alpha=study.best_params["alpha"],
        theta=study.best_params["theta"],
        selection="random",
        verbose=-1,
    )
    remain_data = preprocessed_data_dict["transformed_remain_data"]
    lasso_all.fit(remain_data[:, 1:], remain_data[:, 0])
    if np.sum(np.abs(lasso_all.coef_)) > 0:
        micro_feature_importance = np.abs(lasso_all.coef_)
    else:
        micro_feature_importance = None

    # TODO check if study.best_value is available and at least one trial was completed
    return {
        "shap_values": study.best_trial.user_attrs["shap_values"],
        "macro_feature_importances": study.best_trial.user_attrs["macro_feature_importances"],
        "micro_feature_importance": micro_feature_importance,
    }

    # trial.set_user_attr("shap_values", np.array(shap_values_list))
    # trial.set_user_attr("feature_importances", np.array(coefficients_list))
    # return (
    #     dict(
    #         zip(
    #             study.best_trial.user_attrs["selected_features"],
    #             zip(
    #                 study.best_trial.user_attrs["shap_values"],
    #                 study.best_trial.user_attrs["robustness_selected_features"],
    #             ),
    #         )
    #     ),
    #     study.best_trial.user_attrs["robustness_all_features"],
    #     study.best_trial.user_attrs["feature_importances"],
    # )
