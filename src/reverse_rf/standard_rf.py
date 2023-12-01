import optuna
from optuna.samplers import TPESampler, QMCSampler
import warnings
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, median_absolute_error, roc_auc_score

warnings.filterwarnings("ignore")
# import optuna_study_pruner


def optimize(train_index, validation_index, data_df, meta_data):
    def optuna_objective(trial):
        rf_clf = RandomForestClassifier(
            warm_start=False,
            # max_features=None,
            oob_score=roc_auc_score,
            max_depth=trial.suggest_int("max_depth", 1, 15),
            n_estimators=300,
            random_state=42,
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf", 2, math.floor(len(train_index) / 2)
            ),
        )
        rf_clf.fit(data_df.iloc[train_index, 1:], data_df.loc[train_index, "label"])
        score = rf_clf.oob_score_
        print(score)
        return score

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        direction="maximize",
        sampler=TPESampler(
            # multivariate=True,
            seed=42,
        ),
    )
    if meta_data["parallel"]["cluster"]:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    # terminator = TerminatorCallback()
    study.optimize(
        optuna_objective,
        # n_trials=meta_data["selection_method"]["reverse_trees"]["trials"],
        n_trials=30,
        # callbacks=[terminator],
        timeout=120,
    )

    clf = RandomForestClassifier()
    clf.set_params(**study.best_params)
    clf.fit(data_df.iloc[train_index, 1:], data_df.loc[train_index, "label"])
    predicted_y = clf.predict(data_df.iloc[validation_index, 1:])
    true_y = data_df.loc[validation_index, "label"]
    validation_score = roc_auc_score(true_y, predicted_y)
    return clf.feature_importances_, validation_score
