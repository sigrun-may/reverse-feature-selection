import numpy as np
from optuna import TrialPruned
import optuna
from relaxed_lasso import RelaxedLasso
import celer
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import shap


def select_features(
    preprocessed_data_dict,
    outer_cv_loop,
    meta_data,
    method,
):
    preprocessed_data = preprocessed_data_dict["preprocessed_data_list"]

    def optuna_objective(trial):
        performance_metric_list = []
        params = {"alpha": trial.suggest_uniform("alpha", 0.01, 1.0)}
        if method == "relaxed":
            params["theta"] = trial.suggest_uniform("theta", 0.01, 1.0)

        # # cross validation for HPO
        # for test_data_df, train_data_df, _ in preprocessed_data:
        #     performance_metric, pruned = calculate_performance_metric(
        #         params, train_data_df, test_data_df, "label", method
        #     )
        #     if pruned:
        #         raise TrialPruned()
        #
        #     performance_metric_list.append(performance_metric)

        return calculate_performance_metric_cv(
            params,
            preprocessed_data,
            meta_data,
            trial,
            method
        )

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"standard_lasso_iteration_{outer_cv_loop}",
        direction="maximize",
    )
    if meta_data["parallel"]["cluster"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.optimize(
        optuna_objective,
        n_trials=meta_data["selection_method"]["lasso"]["trials"],
    )

    _, _, micro_coef, _ = calculate_performance_metric(
        study.best_params,
        preprocessed_data_dict["micro_train"],
        preprocessed_data_dict["micro_test"],
        method,
    )
    assert len(micro_coef) == len(preprocessed_data[0][0].columns[1:])
    if np.sum(np.abs(micro_coef)) > 0:
        micro_feature_importance = np.abs(micro_coef)
    else:
        micro_feature_importance = None

    # try:
    #     if not sum(study.best_trial.user_attrs["shap_values"]) > 0:
    #         print("No shap values found")
    # except:  # noqa
    #     return None

    return {
        "shap_values": study.best_trial.user_attrs["shap_values"],
        "macro_feature_importances": study.best_trial.user_attrs[
            "macro_feature_importances"
        ],
        "micro_feature_importance": micro_feature_importance,
    }


def calculate_performance_metric_cv(
    params,
    preprocessed_data,
    meta_data,
    trial,
    method
):
    performance_metric_list = []
    shap_values_list = []
    coefficients_list = []

    # cross validation for HPO
    for test_data_df, train_data_df, _ in preprocessed_data:
        (
            performance_metric,
            mean_shap_values,
            coefficients,
            should_prune,
        ) = calculate_performance_metric(
            params, train_data_df, test_data_df, method
        )
        if should_prune:
            raise TrialPruned()

        performance_metric_list.append(performance_metric)
        shap_values_list.append(mean_shap_values)
        coefficients_list.append(coefficients)

    assert (
        len(shap_values_list) == len(coefficients_list) == len(performance_metric_list)
    )
    trial.set_user_attr("shap_values", np.array(shap_values_list))
    trial.set_user_attr("macro_feature_importances", np.array(coefficients_list))

    return np.mean(performance_metric_list)


def calculate_performance_metric(
    params, train_data_df, test_data_df, method
):
    prune = False

    # prepare train/ test data
    y_train = train_data_df['label'].values.reshape(-1, 1)
    x_train = train_data_df.drop(columns='label')
    x_test = test_data_df.drop(columns='label').values
    y_test = test_data_df['label'].values

    assert x_train.shape[1] >= 1

    # build LASSO model
    if method == "relaxed":
        lasso = RelaxedLasso(
            alpha=params["alpha"],
            theta=params["theta"],
            selection="random",
            verbose=-1,
            # alpha=trial.suggest_discrete_uniform("alpha", 0.001, 1.0,
            # 0.001),
        )
        lasso.fit(x_train.values, y_train)

    elif method == "celer":
        lasso = celer.Lasso(alpha=params["alpha"], verbose=0)
        lasso.fit(x_train.values, y_train)

    elif method == "lasso_sklearn":
        lasso = Lasso(alpha=params["alpha"])
        lasso.fit(np.asfortranarray(x_train), y_train)

    else:
        raise ValueError(
            f"Feature selection method not provided: relaxed, celer and lasso_sklearn are valid options. "
            f"Given method was {method}"
        )

    if "label" in x_train.columns[0]:
        # prune trials if label coefficient is zero or model includes no coefficients
        if lasso.coef_[0] == 0:
            prune = True
    if np.count_nonzero(lasso.coef_) == 0:
        prune = True

    # predict y_test
    performance_metric = r2_score(y_test, lasso.predict(x_test))

    # calculate shap values
    explainer = shap.explainers.Linear(lasso, x_train)
    shap_values = explainer(x_test)
    mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)

    return performance_metric, mean_shap_values, np.abs(lasso.coef_), prune
