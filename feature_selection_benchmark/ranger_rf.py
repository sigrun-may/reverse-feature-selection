# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Standard embedded feature selection with sklearn random forest."""
import concurrent.futures
import math
import warnings

import numpy as np
import optuna
import pandas as pd
import rpy2.robjects as ro
from optuna.samplers import TPESampler
from rpy2.robjects import pandas2ri
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def optimized_ranger_random_forest_importance(data_df: pd.DataFrame, train_indices: np.ndarray, meta_data: dict) -> dict:
    """Calculate importance with the R ranger package with optimized hyperparameters.

    Args:
        data_df: The training data.
        train_indices: The indices of the training split.
        meta_data: The metadata related to the dataset and experiment. Must contain the random state
            for reproducibility of the random forest. Key: "random_state".

    Returns:
        The permutation importances.
    """

    def optuna_objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "num_trees": trial.suggest_int("num_trees", 20, meta_data["max_trees_random_forest"]),
            "mtry": trial.suggest_int("mtry", 1, data_df.shape[1] - 1),
            "regularization_factor": trial.suggest_float("regularization_factor", 0.001, 0.99),
            "seed": meta_data["random_state"],
        }
        oob, _  = ranger_random_forest(data_df, train_indices, params)

        # stop HPO if OOB score (proportion of misclassified observations) is already 0.0
        if math.isclose(oob, 0.0, rel_tol=1e-5):
            trial.study.stop()
        return oob

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        direction="minimize",
        sampler=TPESampler(
            multivariate=True,
            consider_magic_clip=True,
            constant_liar=True,
            seed=48,
        ),
    )
    if not meta_data["verbose_optuna"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.optimize(
        optuna_objective,
        n_trials=meta_data["n_trials_optuna"],
    )

    hyperparameters = study.best_params
    hyperparameters["seed"] = meta_data["random_state"]

    oob_score, feature_importances = ranger_random_forest(data_df, train_indices, hyperparameters)
    return {
        "permutation": feature_importances,
        f"best_params_ranger_permutation": hyperparameters,
        f"oob_score_permutation": oob_score,
    }


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
    # parallelize hyperparameter optimizations
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Submit methods with their specific arguments
        future_sklearn = executor.submit(sklearn_random_forest, data_df, train_indices, meta_data)
        future_permutation = executor.submit(
            optimized_ranger_random_forest_importance, data_df, train_indices, meta_data
        )
        # Collect results as they complete
        result_dict = {}
        for future in concurrent.futures.as_completed([future_sklearn, future_permutation]):
            # merge dictionaries
            result_dict.update(future.result())
    return result_dict


def sklearn_random_forest(data_df: pd.DataFrame, train_indices: np.ndarray, meta_data: dict):
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
            oob_score=roc_auc_score,
            max_depth=trial.suggest_int("max_depth", 1, 15),
            n_estimators=trial.suggest_int("n_estimators", 20, meta_data["max_trees_random_forest"]),
            max_features=trial.suggest_int("max_features", 1, data_df.shape[1] - 1),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, math.floor(len(train_indices) / 2)),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 5),
            min_impurity_decrease=trial.suggest_float("min_impurity_decrease", 0.001, 0.99),
            random_state=meta_data["random_state"],
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        rf_clf.fit(data_df.iloc[train_indices, 1:], data_df.loc[train_indices, "label"])
        score = rf_clf.oob_score_

        # stop HPO if OOB score (AUC) is already 1.0
        if math.isclose(score, 1.0, rel_tol=1e-5):
            trial.study.stop()
        return score

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        direction="maximize",
        sampler=TPESampler(
            multivariate=True,
            consider_magic_clip=True,
            constant_liar=True,
            seed=48,
        ),
    )
    if not meta_data["verbose_optuna"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.optimize(
        optuna_objective,
        n_trials=meta_data["n_trials_optuna"],
    )
    # train random forest with default parameters
    clf = RandomForestClassifier(
        oob_score=roc_auc_score, n_jobs=-1, random_state=meta_data["random_state"], class_weight="balanced_subsample"
    )
    clf.fit(data_df.iloc[train_indices, 1:], data_df.loc[train_indices, "label"])
    gini_feature_importances = clf.feature_importances_
    print("number of selected features (gini default): ", np.sum(gini_feature_importances > 0))

    # train random forest with optimized parameters
    clf.set_params(**study.best_params)
    clf.fit(data_df.iloc[train_indices, 1:], data_df.loc[train_indices, "label"])
    gini_feature_importances_optimized = clf.feature_importances_
    print("number of selected features (gini optimized): ", np.sum(gini_feature_importances_optimized > 0))

    return {
        "gini_impurity": gini_feature_importances_optimized,
        "gini_impurity_default_parameters": gini_feature_importances,
        "train_oob_score_sklearn": clf.oob_score_,
        "best_params_sklearn": study.best_params,
    }


def ranger_random_forest(data_df: pd.DataFrame, train_indices, hyperparameters: dict) -> tuple[float, np.ndarray]:
    """Calculate permutation importance with R.

    Args:
        data_df: The training data.
        train_indices: The indices of the training split.
        hyperparameters: The best hyperparameters.

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
    train_ranger <- function(data, label, max_depth, num_trees, mtry, seed_random_forest, 
                             regularization_factor){

      # ensure label is a factor
      data[[label]] <- as.factor(data[[label]])

      # train the ranger model with optimized parameters
      rf_model <- ranger::ranger(formula = label  ~ .,
                                 data = data,
                                 importance = "permutation",
                                 scale.permutation.importance = TRUE,
                                 regularization.factor = regularization_factor,
                                 min.node.size = 2,
                                 sample.fraction = 0.896551724137931, # 26 from 29 samples
                                 mtry = mtry,
                                 max.depth = max_depth,
                                 num.trees = num_trees,
                                 seed = seed_random_forest,
                                 oob.error = TRUE,
                                 num.threads = 1,
                                )
      # Retrieve the OOB error for debugging
      oob_error <- rf_model$prediction.error
      
      # # print treetype used
      # print("treetype: ")
      # print(rf_model$treetype)

      # get permutation importance
      importance <- rf_model$variable.importance

      result <- c(oob_error, importance)
      return(result)
    }
    """
    # run the R script
    ro.r(r_script)
    # define the function
    train_ranger = ro.globalenv["train_ranger"]
    data = data_df.iloc[train_indices, :]
    assert data.shape[0] == len(train_indices), "Data and train indices do not match."
    assert data.shape[0] < data_df.shape[0], "Training data was not selected."
    assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."

    # call the R function from Python
    result_vector = train_ranger(
        pandas2ri.py2rpy(data),
        "label",
        hyperparameters["max_depth"],
        hyperparameters["num_trees"],
        hyperparameters["mtry"],
        hyperparameters["seed"],
        hyperparameters["regularization_factor"],
    )
    # check if the result_vector is an array of floats
    assert isinstance(result_vector, np.ndarray), "Result vector is not a numpy array."
    assert all(isinstance(x, float) for x in result_vector), "Result vector contains non-float values."
    assert len(result_vector) == data_df.shape[1] # number of features + 1 (oob_error)
    oob_error = result_vector[0]
    feature_importances = result_vector[1:]
    assert isinstance(feature_importances, np.ndarray), "Feature importances are not a numpy array."
    assert isinstance(oob_error, float), "OOB error is not a float."
    assert feature_importances.shape[0] == data_df.shape[1] - 1 # exclude the label column
    # print("number of selected features (permutation_importances): ", np.sum(feature_importances > 0))
    return oob_error, feature_importances

