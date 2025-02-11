import warnings

import lightgbm as lgb
import numpy as np
import shap

warnings.filterwarnings("ignore")


def train_model(parameters, test_data_df, train_data_df, target_feature_name, _):
    # # prepare train/ test data
    # y_train = train_data_df[target_feature].values.reshape(-1, 1)
    # x_train = train_data_df.drop(columns=target_feature)
    # x_test = test_data_df.drop(columns=target_feature).values
    # y_test = test_data_df[target_feature].values
    #
    # train_data = lgb.Dataset(x_train, label=y_train)
    # test_data = lgb.Dataset(x_test, label=y_test)

    # # parameters for model training to combat overfitting
    # parameters = dict(
    #     min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 2, math.floor(x_train.shape[0] / 2)),
    #     lambda_l1=trial.suggest_uniform("lambda_l1", 0.0, 3),
    #     min_gain_to_split=trial.suggest_uniform("min_gain_to_split", 0, 5),
    #     max_depth=trial.suggest_int("max_depth", 2, 20),
    #     bagging_fraction=trial.suggest_uniform("bagging_fraction", 0.1, 1.0),
    #     bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
    #     extra_trees=extra_trees,
    #     objective="binary",
    #     metric="binary_logloss",
    #     boosting_type="rf",
    #     verbose=-1,
    # )

    # # num_leaves must be smaller than 2^max_depth
    # # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree
    # max_num_leaves = 2 ** parameters["max_depth"] - 1
    # max_num_leaves = min(max_num_leaves, 90)
    # parameters["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)

    # model = lgb.train(
    #     parameters,
    #     train_data,
    #     valid_sets=[test_data],
    #     callbacks=[lgb.record_evaluation({})],  # stop verbose
    # )
    prune = False

    # prepare train/ test data
    y_train = train_data_df[target_feature_name].values.reshape(-1, 1)
    x_train = train_data_df.drop(columns=target_feature_name)
    x_test = test_data_df.drop(columns=target_feature_name).values
    y_test = test_data_df[target_feature_name].values

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test)

    assert x_train.shape[1] >= 1

    # build model
    model = lgb.train(
        parameters,
        train_data,
        valid_sets=[test_data],
        callbacks=[lgb.record_evaluation({})],  # stop verbose
    )

    if "label" in x_train.columns[0]:
        # prune trials if label coefficient is zero or model includes no coefficients
        if model.feature_importance(importance_type="gain")[0] == 0:
            prune = True
    if np.count_nonzero(model.feature_importance(importance_type="gain")) == 0:
        prune = True

    return model, prune


def get_shap_list(model, x_test):
    # calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x_test)
    raw_shap_values = np.abs(shap_values.values)[:, :, 0]
    return np.sum(raw_shap_values, axis=0)
