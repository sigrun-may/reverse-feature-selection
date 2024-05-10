# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Standard embedded feature selection with sklearn random forest."""

import math
import warnings

import numpy as np
import optuna
import shap
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def optimize(train_indices, data_df, meta_data):
    """
    Optimize the hyperparameters of a random forest classifier using optuna.

    Args:
        train_indices: The indices of the training split.
        data_df: The training data.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        The feature importances, the summed shap values, and the out-of-bag (OOB) score.
    """

    def optuna_objective(trial):
        rf_clf = RandomForestClassifier(
            oob_score=roc_auc_score,
            max_depth=trial.suggest_int("max_depth", 1, 15),
            n_estimators=300,
            random_state=meta_data.get("random_state", None),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, math.floor(len(train_indices) / 2)),
            min_impurity_decrease=trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
            n_jobs=-1,
        )
        rf_clf.fit(data_df.iloc[train_indices, 1:], data_df.loc[train_indices, "label"])
        score = rf_clf.oob_score_
        return score

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        direction="maximize",
        sampler=TPESampler(
            multivariate=True,
            seed=42,
        ),
    )
    if not meta_data["verbose_optuna"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    # terminator = TerminatorCallback()
    study.optimize(
        optuna_objective,
        n_trials=meta_data["n_trials_optuna"],
        # callbacks=[terminator],
        # timeout=120,
    )

    clf = RandomForestClassifier(
        oob_score=roc_auc_score,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    clf.set_params(**study.best_params)
    clf.fit(data_df.iloc[train_indices, 1:], data_df.loc[train_indices, "label"])
    # # L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.
    # p_importances = permutation_importance(clf, X=data_df.iloc[train_indices, 1:], y=data_df.loc[train_indices, "label"],
    #                                        n_repeats=5, random_state=42, n_jobs=-1)

    # calculate shap values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(data_df.iloc[train_indices, 1:])
    sum_shap_values = np.zeros(data_df.shape[1] - 1)
    for values_per_sample in shap_values:
        positive_shap_values = np.abs(values_per_sample[:, 0])
        sum_shap_values += positive_shap_values

    feature_importances = clf.feature_importances_
    print("number of selected features: ", np.sum(feature_importances > 0))

    return clf.feature_importances_, sum_shap_values, clf.oob_score_
