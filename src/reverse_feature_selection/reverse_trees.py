import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna import TrialPruned
from sklearn.metrics import r2_score
from relaxed_lasso import RelaxedLasso
import reverse_selection
import celer
from sklearn.linear_model import Lasso
import warnings
import math
from lightgbm import LGBMRegressor as lgb
from optuna import TrialPruned, pruners
import tree_model
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

        # parameters for model training to combat overfitting
        _, train_data_df, _ = preprocessed_data[0]
        parameters = dict(
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, math.floor(train_data_df.shape[0] / 2)),
            lambda_l1=trial.suggest_uniform("lambda_l1", 0.0, 3),
            min_gain_to_split=trial.suggest_uniform("min_gain_to_split", 0, 5),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            bagging_fraction=trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            extra_trees=(method == 'extra_trees'),
            objective="regression",
            boosting_type="rf",
            verbose=0,
        )
        # num_leaves must be smaller than 2^max_depth
        # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree
        max_num_leaves = 2 ** parameters["max_depth"] - 1
        max_num_leaves = min(max_num_leaves, 90)
        parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)

        return reverse_selection.calculate_performance_metric_cv(
            parameters,
            target_feature_name,
            preprocessed_data,
            meta_data,
            method,
            tree_model.train_model,
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
    study.optimize(
        optuna_objective,
        n_trials=meta_data["selection_method"]["reverse_trees"]["trials"],
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
        tree_model.train_model,
        include_label=False,
        deselected_features=deselected_features,
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


# def calculate_performance_metric(parameters, train_data_df, test_data_df, target_feature_name, _):
#     prune = False
#
#     # prepare train/ test data
#     y_train = train_data_df[target_feature_name].values.reshape(-1, 1)
#     x_train = train_data_df.drop(columns=target_feature_name)
#     x_test = test_data_df.drop(columns=target_feature_name).values
#     y_test = test_data_df[target_feature_name].values
#
#     train_data = lgb.Dataset(x_train, label=y_train)
#     test_data = lgb.Dataset(x_test, label=y_test)
#
#     assert x_train.shape[1] >= 1
#
#     # build model
#     model = lgb.train(
#         parameters,
#         train_data,
#         valid_sets=[test_data],
#         callbacks=[lgb.record_evaluation({})],  # stop verbose
#     )
#
#     if "label" in x_train.columns[0]:
#         # prune trials if label coefficient is zero or model includes no coefficients
#         if model.feature_importance(importance_type='gain')[0] == 0:
#             prune = True
#     if np.count_nonzero(model.feature_importance(importance_type='gain')) == 0:
#         prune = True
#
#     performance_metric = model.score(x_test, y_test)
#     # # predict y_test
#     # performance_metric = r2_score(y_test, model.predict(x_test))
#
#     return performance_metric, prune
