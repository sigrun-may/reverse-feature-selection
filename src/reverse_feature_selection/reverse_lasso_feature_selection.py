import math

import numpy as np
import pandas as pd
import optuna
from optuna import TrialPruned
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score
import datetime
import settings
import celer
#


def calculate_adjusted_r2(y, predicted_y, number_of_coefficients):
    # calculate the coefficient of determination, r2:
    # The proportion of the variance in the dependent variable that is predictable from the independent variable(s).
    # It provides selected_feature_subset_list measure of how well observed outcomes are replicated by the model, based on the proportion of total
    # variation of outcomes explained by the model
    r2 = r2_score(y, predicted_y)
    # print('r2:', r2)

    samplesize = len(y)
    # print('number_of_coefficients', number_of_coefficients)
    # adjusted_r2 = 1 - (1 - r2) * (samplesize - 1) / (samplesize - (np.max(number_of_coefficients)) - 1)
    adjusted_r2 = 1 - (1 - r2) * (samplesize - 1) / (
        samplesize - (np.median(number_of_coefficients)) - 1
    )
    # print('adjusted_r2: ', adjusted_r2)

    # R2 gibt den Prozentsatz der Streuung der Antwortvariablen an, der durch das Modell erklärt wird. Der Wert wird wie
    # folgt berechnet: 1 minus das Verhältnis zwischen der Summe der quadrierten Fehler (Streuung, die durch das Modell
    # nicht erklärt wird) zur Gesamtsumme der Quadrate (Gesamtstreuung im Modell).

    # Adj r2 = 1-(1-R2)*(n-1)/(n-p-1)
    # assume n = number of sample size , p = number of independent variables
    # Je höher das korrigierte R², desto besser passt das Modell auf die Daten.
    return adjusted_r2
    # return r2


def calculate_r2_adjusted(
    alpha,
    target_feature_name,
    deselected_features,
    data_dict,
    include_label,
    trial,
    correlation_threshold,
):
    predicted_y = []
    true_y = []
    number_of_coefficients = []
    label_coefficients = []
    count_no_label_included = 0

    feature_names = data_dict["feature_names"].values
    transformed_data = data_dict["transformed_data"]

    start = datetime.datetime.now()

    # cross validation for the optimization of alpha
    for fold_index, (test, train, train_correlation_matrix_complete) in enumerate(
        transformed_data
    ):

        # remove irrelevant features from train_correlation_matrix
        train_correlation_matrix = train_correlation_matrix_complete.drop(
            labels=deselected_features,
            axis=0,
            inplace=False,
        )
        train_correlation_matrix.drop(
            labels=deselected_features,
            axis=1,
            inplace=True,
        )
        assert (
            train_correlation_matrix_complete.shape[1] - len(deselected_features)
            == train_correlation_matrix.shape[1]
        )
        assert (
            train_correlation_matrix_complete.shape[0] - len(deselected_features)
            == train_correlation_matrix.shape[0]
        )

        # remove irrelevant features from test and train data
        train_data_df = pd.DataFrame(train, columns=feature_names)
        train_data_df.drop(columns=deselected_features, inplace=True)
        assert train_data_df.shape[1] == train.shape[1] - len(deselected_features)

        test_data_df = pd.DataFrame(test, columns=feature_names)
        test_data_df.drop(columns=deselected_features, inplace=True)
        assert test_data_df.shape[1] == test.shape[1] - len(deselected_features)

        # print('drop', datetime.datetime.now() - start)
        start = datetime.datetime.now()

        # How Correlations Influence Lasso Prediction, April 2012IEEE Transactions on Information Theory 59(3)
        # DOI: 10.1109/TIT.2012.2227680

        # find features correlated to the target_feature from test/ train data
        correlated_features = [
            index
            for index, value in train_correlation_matrix[target_feature_name].items()
            if abs(value) > correlation_threshold
        ]
        if target_feature_name in correlated_features:
            correlated_features.remove(target_feature_name)

        # print('get correlations', datetime.datetime.now() - start)
        start = datetime.datetime.now()

        if not include_label:
            # append label to the list of features to remove
            correlated_features.append("label")

        # check if train would keep at least one feature
        if train_data_df.shape[1] - len(correlated_features) == 0:
            # keep the feature with the lowest correlation to the target feature
            correlated_features.remove(np.min(np.abs(correlated_features)))

        # remove features correlated to the target_feature from test/ train data and the label if it is not included
        train_data_df.drop(columns=correlated_features, inplace=True)
        test_data_df.drop(columns=correlated_features, inplace=True)
        assert train_data_df.shape[1] == train.shape[1] - len(
            deselected_features
        ) - len(correlated_features)
        assert train_data_df.shape[1] > 0

        assert test_data_df.shape[1] == test.shape[1] - len(deselected_features) - len(
            correlated_features
        )
        assert test_data_df.shape[1] > 0

        # prepare train/ test data
        y_train = train_data_df[target_feature_name].values.reshape(-1, 1)
        x_train = train_data_df.drop(columns=target_feature_name)
        x_test = test_data_df.drop(columns=target_feature_name).values
        true_y.extend(test_data_df[target_feature_name].values)

        # print('prepare', datetime.datetime.now() - start)
        start = datetime.datetime.now()

        # build LASSO model
        lasso = celer.Lasso(alpha=alpha, fit_intercept=True, positive=False, prune=True)
        lasso.fit(x_train, y_train)

        # model = cuml.Lasso(fit_intercept = False, normalize = True)
        # model.fit(X_train, y_train).predict(X_test)

        # sklearn lasso
        # lasso = Lasso(alpha = alpha, fit_intercept = True, positive = False)
        # lasso.fit(np.asfortranarray(x_train), y_train)

        # print('predict', datetime.datetime.now() - start)
        start = datetime.datetime.now()

        # save number of non zero coefficients to calculate r2 adjusted
        number_of_coefficients.append(sum(lasso.coef_ != 0.0))
        # print(lasso.coef_[np.nonzero(lasso.coef_)])

        if include_label:
            # prune trials without any coefficients in the model
            # if number_of_coefficients[-1] == 0:
            if sum(lasso.coef_ != 0.0) == 0:
                assert sum(lasso.coef_ != 0.0) == number_of_coefficients[-1]
                raise TrialPruned()

            # check label coefficient
            label_coefficient = lasso.coef_[0]
            label_coefficients.append(label_coefficient)
            # if label_coefficient > 0:
            #     print('#################################################################################################')
            #     print('label:', label_coefficient)

            # prune trials if one quarter of all label coefficients is zero
            if label_coefficient == 0:
                raise TrialPruned()
                # count_no_label_included += 1
                # if count_no_label_included > (len(transformed_data) / 4):
                #     raise TrialPruned()

        assert len(lasso.coef_) == x_train.shape[1]

        # predict y_test
        predicted_y.extend(lasso.predict(x_test))

    if include_label:
        trial.set_user_attr("label_coefficients", label_coefficients)

    adjusted_r2 = calculate_adjusted_r2(true_y, predicted_y, number_of_coefficients)

    # print('the rest', datetime.datetime.now() - start)

    return adjusted_r2


def optimize(
    transformed_test_train_splits_dict,
    target_feature_name,
    correlation_threshold,
    deselected_features,
    outer_cv_loop,
):
    """Optimize regularization parameter alpha for lasso regression."""

    def optuna_objective(trial):

        # if "random" in target_feature_name or "pseudo" in target_feature_name:
        #     trial.study.stop()

        if trial.number >= 10:
            try:
                trial.study.best_value
            except:
                # print('no results for more than 10 trials')
                # print(target_feature_name)
                trial.study.stop()

        #  adapt TPE sampler to suggest new parameter, when suggested parameter is repeated
        alphas_history = []
        for _trial in study.trials:
            if _trial.user_attrs:
                alphas_history.append(_trial.user_attrs["alpha"])
            else:
                continue

        # alpha = trial.suggest_discrete_uniform("alpha", 0.01, 1.0, 0.01)
        # print(alphas_history)
        alpha = trial.suggest_int("alpha", 1, 101, log=True) / 100
        if alpha in alphas_history:
            # print('alpha', alpha)
            alpha = trial.suggest_int("beta", 1, 101, log=True) / 100
            # print('alpha_beta', alpha)
        if alpha in alphas_history:
            # print('alpha', alpha)
            alpha = trial.suggest_int("gamma", 1, 101, log=True) / 100
            # print('alpha_gamma', alpha)

        # prune study when aplha is repeated
        if alpha in alphas_history:
            # print("repeated alpha: ", alpha, "in trial ", trial.number)
            raise TrialPruned()
        # alphas_history.append(alpha)

        # save calculated alpha
        trial.set_user_attr("alpha", alpha)

        r2_adj = calculate_r2_adjusted(
            alpha,
            target_feature_name,
            deselected_features,
            transformed_test_train_splits_dict,
            include_label=True,
            trial=trial,
            correlation_threshold=correlation_threshold,
        )
        return r2_adj

    # try:
    #     optuna.study.delete_study(f"{target_feature_name}_iteration_{outer_cv_loop}", storage = "sqlite:///optuna_db.db")
    # except:
    #     print("new study")

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"{target_feature_name}_iteration_{outer_cv_loop}",
        # study_name = f"{target_feature_name}_iteration_{outer_cv_loop}",
        direction="maximize",
        sampler=TPESampler(
            multivariate=False,
            n_startup_trials=3,
            consider_magic_clip=True,
            constant_liar=True,
        ),
    )
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        # n_trials = 40,
        n_trials=20,
        n_jobs=1,
    )

    # check if study.best_value is available and at least one trial was completed
    try:
        study.best_value
        # if study.best_value > 0:
        #     fig = optuna.visualization.plot_optimization_history(study)
        #     fig.show()
    except:
        return 0, 0, None

    # Was the label included in the best model?
    if np.sum(np.abs(study.best_trial.user_attrs["label_coefficients"])) > 0:
        # Select the best regularization parameter:
        # Consider the case when the same alpha was proposed twice by TPE
        # and therefore an alternative parameter was selected to calculate the trial.
        if "gamma" in study.best_params.keys():
            best_alpha = study.best_params["gamma"]
            # print(study.best_params)
            # print(best_alpha)
        elif "beta" in study.best_params.keys():
            best_alpha = study.best_params["beta"]
            # print(study.best_params)
            # print(best_alpha)
        else:
            best_alpha = study.best_params["alpha"]

        # calculate r2_adjusted adjusted without the label in the training data for the same alpha
        r2_adjusted = calculate_r2_adjusted(
            best_alpha,
            target_feature_name,
            deselected_features,
            transformed_test_train_splits_dict,
            include_label=False,
            trial=None,
            correlation_threshold=correlation_threshold,
        )
        return (
            r2_adjusted,
            study.best_value,
            study.best_trial.user_attrs["label_coefficients"],
        )
    else:
        return (
            math.inf,  # increase r2_adjusted_unlabeled to falsify r2_adjusted_unlabeled < r2_adjusted
            study.best_value,
            study.best_trial.user_attrs["label_coefficients"],
        )


def select_features(transformed_test_train_splits_dict, outer_cv_loop, correlation_threshold):

    # calculate relevance for each feature
    deselected_features_list = []
    selected_features_dict = {}
    for target_feature_name in transformed_test_train_splits_dict["feature_names"]:

        # exclude the label
        if target_feature_name == "label":
            continue

        # get performance of target feature
        (
            r2_adjusted_unlabeled,
            r2_adjusted,
            label_coefficients_list,
        ) = optimize(
            transformed_test_train_splits_dict,
            target_feature_name,
            correlation_threshold,
            deselected_features_list,
            outer_cv_loop=outer_cv_loop,
        )

        if (r2_adjusted > 0) and (r2_adjusted_unlabeled < r2_adjusted):
            # select feature
            selected_features_dict[target_feature_name] = (
                r2_adjusted_unlabeled,
                r2_adjusted,
                label_coefficients_list,
            )
        else:
            # exclude irrelevant feature from training data
            deselected_features_list.append(target_feature_name)

    return selected_features_dict
