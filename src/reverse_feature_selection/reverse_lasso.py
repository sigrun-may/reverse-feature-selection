import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna import TrialPruned

from relaxed_lasso import RelaxedLasso
import reverse_selection
# import celer
from sklearn.linear_model import Lasso
import warnings

warnings.filterwarnings("ignore")


# import optuna_study_pruner


def optimize(
    preprocessed_data,
    target_feature_name,
    deselected_features,
    outer_cv_loop,
    meta_data,
    method,
):
    """Optimize regularization parameter alpha for lasso regression."""

    def optuna_objective(trial):
        # optuna_study_pruner.study_no_trial_completed_pruner(trial, warm_up_steps=20)
        # optuna_study_pruner.study_no_improvement_pruner(
        #     trial,
        #     epsilon=0.001,
        #     warm_up_steps=30,
        #     number_of_similar_best_values=5,
        #     threshold=0.1,
        # )

        # optuna_study_pruner.insufficient_results_study_pruner(
        #     trial, warm_up_steps=20, threshold=0.05
        # )
        params = {"alpha": trial.suggest_uniform("alpha", 0.01, 1.0)}
        if method == "relaxed":
            params["theta"] = trial.suggest_uniform("theta", 0.01, 1.0)

        return reverse_selection.calculate_performance_metric_cv(
            params,
            target_feature_name,
            preprocessed_data,
            meta_data,
            method,
            calculate_performance_metric,
            include_label=True,
            deselected_features=deselected_features,
        )

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
    if meta_data["parallel"]["cluster"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        n_trials=meta_data["selection_method"]["reverse_lasso"]["trials"],
    )
    # if study.best_value > 0:
    #     fig = optuna.visualization.plot_optimization_history(study)
    #     fig.show()

    # check if study.best_value is available and at least one trial was completed
    try:
        if study.best_value <= 0:  # Was r2 adjusted greater than zero?
            return 0, 0
    except:  # noqa
        return 0, 0

    # calculate r2 without the label in the training data for the same alpha
    r2 = reverse_selection.calculate_performance_metric_cv(
        study.best_params,
        target_feature_name,
        preprocessed_data,
        meta_data,
        method,
        calculate_performance_metric,
        include_label=False,
        deselected_features=None,
    )
    print(
        target_feature_name,
        study.best_value,
        r2,
    )
    return (
        r2,
        study.best_value,
    )


def calculate_performance_metric(params, _, train_data_df, target_feature_name, method):
    prune = False

    # prepare train data
    y_train = train_data_df[target_feature_name].values.reshape(-1, 1)
    x_train = train_data_df.drop(columns=target_feature_name)

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

    # elif method == "celer":
    #     lasso = celer.Lasso(alpha=params["alpha"], verbose=0)
    #     lasso.fit(x_train.values, y_train)

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

    # # predict y_test
    # performance_metric = r2_score(y_test, lasso.predict(x_test))

    return lasso, prune
    # return performance_metric, prune
