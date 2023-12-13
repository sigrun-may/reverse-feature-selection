import numpy as np
import optuna
import shap
from optuna.samplers import TPESampler
import warnings
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# import optuna_study_pruner


def optimize(train_index, validation_index, data_df, meta_data):
    def optuna_objective(trial):
        rf_clf = RandomForestClassifier(
            warm_start=False,
            oob_score=roc_auc_score,
            max_depth=trial.suggest_int("max_depth", 1, 15),
            n_estimators=300,
            random_state=42,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, math.floor(len(train_index) / 2)),
            min_impurity_decrease=trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
            n_jobs=-1,
        )
        rf_clf.fit(data_df.iloc[train_index, 1:], data_df.loc[train_index, "label"])
        score = rf_clf.oob_score_
        print(score)
        return score

    # TODO move direction to settings.toml
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
    # TODO move n_trials to settings.toml
    study.optimize(
        optuna_objective,
        n_trials=meta_data["n_trials_optuna"],
        # callbacks=[terminator],
        timeout=120,
    )

    clf = RandomForestClassifier(
        oob_score=roc_auc_score,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    clf.set_params(**study.best_params)
    clf.fit(data_df.iloc[train_index, 1:], data_df.loc[train_index, "label"])
    # # L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.
    # p_importances = permutation_importance(clf, X=data_df.iloc[train_index, 1:], y=data_df.loc[train_index, "label"],
    #                                        n_repeats=5, random_state=42, n_jobs=-1)
    predicted_y = clf.predict(data_df.iloc[validation_index, 1:])
    true_y = data_df.loc[validation_index, "label"]
    validation_score = roc_auc_score(true_y, predicted_y)
    print("validation_score", validation_score, "oob", clf.oob_score_)

    # calculate shap values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(data_df.iloc[train_index, 1:])
    assert len(shap_values) == 2
    assert len(shap_values[0]) == len(shap_values[1]) == len(data_df.iloc[train_index, 1:])

    # sum shap values
    mean_shap_values = np.abs(shap_values[0]).mean(axis=0)
    assert len(mean_shap_values) == len(data_df.columns) - 1

    return clf.feature_importances_, mean_shap_values, validation_score, clf.oob_score_
