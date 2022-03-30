from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
)
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import numpy as np


def get_prediction(feature, x, y):
    predicted_y = []
    true_values = []
    true_y = []
    predicted_y_auc = []
    metric_list = []

    sample_loo_test = LeaveOneOut()
    for loo_index, (train_index, test_index) in enumerate(sample_loo_test.split(x)):
        # print(loo_index)
        x_train = x.iloc[train_index, :]
        y_train = y.iloc[train_index]
        x_test = x.iloc[test_index, :]
        y_test = y.iloc[test_index]

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=420)
        estimator = clf.fit(x_train[feature].values.reshape(-1, 1), y_train)

        prediction = estimator.predict_proba(x_test[feature].values.reshape(-1, 1))
        prediction_auc = estimator.predict(x_test[feature].values.reshape(-1, 1))
        predicted_y_auc.extend(prediction_auc)

        # print(prediction)
        # predicted_y.append(prediction[0][0])
        # predicted_y.append(prediction[0][1])
        predicted_y.append(prediction[0])
        # print(prediction)
        # print('true value')
        # print(y_test)

        # true_values.append(clf.classes_[0])
        # true_values.append(clf.classes_[1])
        true_values.append(clf.classes_)
        true_y.extend(y_test)

        ll = log_loss(y_test, prediction, labels=clf.classes_)
        # ll=roc_auc_score(y_test, prediction)
        # print(ll)
        metric_list.append(ll)
        # print(log_loss_list)

        # print(matthews_corrcoef(true_values, predicted_y))
        # true_values.append(y_test[0])
    # scores = cross_val_score(clf, x, y, cv=10)
    # print(scores)

    test_auc = roc_auc_score(true_y, predicted_y_auc)
    #
    # # test_f1 = f1_score(true_values, predicted_y, labels = [0, 1])
    # print(test_auc)
    # return true_values, predicted_y
    # print('-------------------')
    # print(predicted_y)
    # print(true_values)
    # print(true_values)
    # print(predicted_y)
    # print(matthews_corrcoef(true_values, predicted_y))
    # matthews = matthews_corrcoef(y_true = true_values, y_pred = predicted_y)
    # brier = brier_score_loss(true_values, predicted_y)
    # # print(matthews)
    # return brier
    # return roc_auc_score(true_values, predicted_y)
    # print(log_loss(true_values, predicted_y))
    # if np.mean(metric_list) < 0.2:
    #     print(feature)
    #     return feature
    # else:
    #     return 'NAN'
    if test_auc > 0.7:
        print(feature)
        return feature
    else:
        return "NAN"
    # return test_auc
    # return np.mean(metric_list)

    # print('final result', get_prediction())


def get_scores(data_df):
    # data = data_df.to_numpy()
    # y is the target feature to calculate the feature importance
    y = data_df.iloc[:, 0]
    x = data_df.iloc[:, 1:]

    # progress_bar = tqdm(total = x.shape[1], postfix='highest
    # evaluation_metric: ?')
    start = time.time()
    number_of_features = x.shape[1]
    scores = Parallel(n_jobs=1, prefer="threads", verbose=5)(delayed(get_prediction)(feature, x, y) for feature in x)
    # return scores
    # return [x for x in scores if "NAN" not in x]
    return [x for x in scores if "NAN" not in x]
    # true_values, predicted_y = zip(*scores)
    #
    # print(max(scores))
    # print(min(scores))
    # return list(zip(data_df.columns[1:], scores)), max(scores), min(scores)
    #
    # print('duration', time.time() - start)
    #
    # # joblib.dump(scores, "pickled_files/mcc_decision_tree_colon.pkl.gz", compress=('gzip', 3))
    # # joblib.dump(scores, "pickled_files/true_y_and_predicted_y_proba_decision_tree_hip.pkl.gz", compress=('gzip', 3))
    # joblib.dump(scores, "pickled_files/logloss_tree_colon.pkl.gz", compress=('gzip', 3))
