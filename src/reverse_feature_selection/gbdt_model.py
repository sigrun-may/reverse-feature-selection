import lightgbm as lgb
import math
import shap
import numpy as np


def train_model(test_data_df, train_data_df, target_feature, trial):
    # prepare train/ test data
    y_train = train_data_df[target_feature].values.reshape(-1, 1)
    x_train = train_data_df.drop(columns=target_feature)
    x_test = test_data_df.drop(columns=target_feature).values
    y_test = test_data_df[target_feature].values

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test)

    # parameters for model training to combat overfitting
    parameters = dict(
        min_data_in_leaf=trial.suggest_int(
            "min_data_in_leaf", 2, math.floor(x_train.shape[0] / 2)
        ),
        lambda_l1=trial.suggest_uniform("lambda_l1", 0.0, 3),
        min_gain_to_split=trial.suggest_uniform("min_gain_to_split", 0, 5),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        bagging_fraction=trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
        extra_trees=False,
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        verbose=-1,
        num_iterations=trial.suggest_int("num_iterations", 3, 30),
    )

    # num_leaves must be smaller than 2^max_depth
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree
    max_num_leaves = 2 ** parameters["max_depth"] - 1
    max_num_leaves = min(max_num_leaves, 90)
    parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)

    model = lgb.train(
        parameters,
        train_data,
        valid_sets=[test_data],
        callbacks=[lgb.record_evaluation({})],  # stop verbose
    )
    return model


def get_shap_list(model, x_test):
    # calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_test)
    raw_shap_values = np.abs(shap_values.values)[:, :, 0]
    return np.sum(raw_shap_values, axis=0)
