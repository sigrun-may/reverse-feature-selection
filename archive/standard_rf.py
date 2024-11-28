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
import pandas as pd
import rpy2.robjects as ro
from optuna.samplers import TPESampler
from rpy2.robjects import pandas2ri
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")


def calculate_feature_importance(data_df: pd.DataFrame, train_indices: np.ndarray, meta_data: dict) -> dict:
    """Optimize the hyperparameters of a random forest classifier using optuna.

    Args:
        data_df: The training data.
        train_indices: The indices of the training split.
        meta_data: The metadata related to the dataset and experiment.

    Returns:
        The feature importances, the summed shap values, the permutation importance, the out-of-bag (OOB) score
        and the best hyperparameter values.
    """

    def optuna_objective(trial):
        rf_clf = RandomForestClassifier(
            oob_score=accuracy_score,
            max_depth=trial.suggest_int("max_depth", 1, 15),
            n_estimators=trial.suggest_int("n_estimators", 20, 3000),
            max_features=trial.suggest_int("max_features", 1, data_df.shape[1] - 1),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, math.floor(len(train_indices) / 2)),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 5),
            min_impurity_decrease=trial.suggest_float("min_impurity_decrease", 0.1, 0.5),
            random_state=meta_data["random_state"],
            class_weight="balanced_subsample",
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
    # train random forest with default parameters
    clf = RandomForestClassifier(
        oob_score=roc_auc_score, n_jobs=-1, random_state=meta_data["random_state"], class_weight="balanced_subsample"
    )
    clf.fit(data_df.iloc[train_indices, 1:], data_df.loc[train_indices, "label"])
    gini_feature_importances = clf.feature_importances_
    print("number of selected features (gini): ", np.sum(gini_feature_importances > 0))

    # train random forest with optimized parameters
    clf.set_params(**study.best_params)
    clf.fit(data_df.iloc[train_indices, 1:], data_df.loc[train_indices, "label"])
    gini_feature_importances_optimized = clf.feature_importances_
    print("number of selected features (gini optimized): ", np.sum(gini_feature_importances_optimized > 0))

    # # calculate shap values
    # explainer = shap.TreeExplainer(clf)
    # shap_values = explainer.shap_values(data_df.iloc[train_indices, 1:])
    # sum_shap_values = np.zeros(data_df.shape[1] - 1)
    # for values_per_sample in shap_values:
    #     positive_shap_values = np.abs(values_per_sample[:, 0])
    #     sum_shap_values += positive_shap_values

    # calculate permutation importance with R applying optimized parameters
    # L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.
    permutation_importances = ranger_ramdom_forest_importance(
        data_df, train_indices, study.best_params, meta_data["random_state"]
    )
    return {
        "gini_impurity": gini_feature_importances_optimized,
        "gini_impurity_default_parameters": gini_feature_importances,
        # "summed_shap_values": sum_shap_values,
        "permutation_importance": permutation_importances,
        "train_oob_score": clf.oob_score_,
        "best_params": study.best_params,
    }


def ranger_ramdom_forest_importance(
    data_df: pd.DataFrame, train_indices, best_params: dict, random_state: int
) -> np.ndarray:
    """Calculate permutation importance with R.

    Args:
        data_df: The training data.
        train_indices: The indices of the training split.
        best_params: The best hyperparameters.
        random_state: The random state for reproducibility.
        importance_type: The type of importance to calculate. Either "permutation" or "impurity_corrected".

    Returns:
        The permutation importances.
    """
    # calculate permutation importance with R
    # activate the pandas2ri conversion
    pandas2ri.activate()

    # R script to train a ranger model and compute permutation importance
    r_script = """
    library(ranger)

    # train a ranger model with optimized hyperparameters
    train_ranger <- function(data, label, max_depth, num_trees, mtry, min_node_size, seed_random_forest){
      # ensure label is a factor
      data[[label]] <- as.factor(data[[label]])

      # train the ranger model with optimized parameters
      rf_model <- ranger::ranger(formula = as.formula(paste(label, "~ .")), 
                                 data = data, 
                                 importance = 'permutation',
                                 min.node.size = min_node_size,
                                 mtry = mtry,
                                 max.depth = max_depth,
                                 num.trees = num_trees,
                                 seed = seed_random_forest,
                                 oob.error = TRUE,
                                )
      # Retrieve the OOB error for debugging
      oob_error <- rf_model$prediction.error
      print("oob_error ranger:")
      print(oob_error)

      # get permutation importance
      importance <- rf_model$variable.importance
      return(importance)
    }
    """
    # run the R script
    ro.r(r_script)
    # define the function
    train_ranger = ro.globalenv["train_ranger"]
    data = data_df.iloc[train_indices, :]
    assert data.shape[0] == len(train_indices), "Data and train indices do not match."
    assert data.shape[0] < data_df.shape[0], "Training data was not selected."

    # transform the params dictionary to single hyperparameters
    max_depth = best_params["max_depth"]
    num_trees = best_params["n_estimators"]
    mtry = best_params["max_features"]
    min_node_size = best_params["min_samples_split"]

    # call the R function from Python
    # function(data, label, max_depth, num_trees, mtry, min_node_size, seed_random_forest)
    permutation_importances = train_ranger(
        pandas2ri.py2rpy(data), "label", max_depth, num_trees, mtry, min_node_size, random_state
    )
    print("number of selected features (permutation_importances): ", np.sum(permutation_importances > 0))
    return permutation_importances
