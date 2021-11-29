from ctypes import Union
from typing import Dict, List, Tuple

from optuna.samplers import TPESampler
from optuna import TrialPruned
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import celer
import shap

import numpy as np
from sklearn.metrics import r2_score


def calculate_adjusted_r2(y, predicted_y, number_of_coefficients):
    # calculate the coefficient of determination, r2:
    # The proportion of the variance in the dependent variable that is predictable from the independent variable(s).
    # It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total
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


def select_features(
    preprocessed_data_dict: Dict[str, List[Union[Tuple[np.array], str]]],
) -> Dict[str, float]:
    pass

    """Optimize regularization parameter alpha for lasso regression."""

    def optuna_objective(trial):

        # if "random" in target_feature_name or "pseudo" in target_feature_name:
        #     trial.study.stop()

        if trial.number >= 10:
            try:
                print(trial.study.best_value)
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
            print("repeated alpha: ", alpha, "in trial ", trial.number)
            raise TrialPruned()
        # alphas_history.append(alpha)

        # save calculated alpha
        trial.set_user_attr("alpha", alpha)

        predicted_y = []
        true_y = []
        number_of_coefficients = []
        coefficients = []
        all_shap_values = []
        sum_of_all_shap_values = []

        feature_names = preprocessed_data_dict["feature_names"]
        transformed_data = preprocessed_data_dict["transformed_data"]

        # cross validation for the optimization of alpha
        for fold_index, (test, train, train_correlation_matrix_complete) in enumerate(
            transformed_data
        ):
            # TODO pandas notwendig?
            train_data_df = pd.DataFrame(train, columns=feature_names)
            test_data_df = pd.DataFrame(test, columns=feature_names)

            # prepare train/ test data
            y_train = train_data_df["label"].values.reshape(-1, 1)
            x_train = train_data_df.drop(columns="label")
            x_test = test_data_df.drop(columns="label").values
            true_y.extend(test_data_df["label"].values)

            # build LASSO model
            lasso = celer.Lasso(
                alpha=alpha, fit_intercept=True, positive=False, prune=True
            )
            lasso.fit(x_train, y_train)
            coefficients.append(lasso.coef_)

            # sklearn lasso
            # lasso = Lasso(alpha = alpha, fit_intercept = True, positive = False)
            # lasso.fit(np.asfortranarray(x_train), y_train)

            number_of_coefficients.append(sum(lasso.coef_ != 0.0))

            # explain the model's predictions using SHAP
            explainer = shap.explainers.Linear(lasso, x_train)
            shap_values = explainer(x_train)
            # shap.plots.heatmap(shap_values)
            # plt.show()

            cumulated_shap_values = np.stack(np.abs(shap_values.values), axis=0)
            all_shap_values.append(cumulated_shap_values)
            sum_of_all_shap_values.append(np.sum(cumulated_shap_values, axis=0))

            # TODO unterschied coeff zu stacked SHAP

            # predict y_test
            predicted_y_test = lasso.predict(x_test)
            # predicted_y.append(predicted_y_test[0])
            predicted_y.extend(predicted_y_test)

        feature_idx = np.sum(np.array(sum_of_all_shap_values), axis=0)
        selected_features = feature_names[feature_idx.nonzero()]
        assert len(selected_features) == feature_idx.nonzero()[0].size
        # trial.set_user_attr("label_coefficients", coefficients)
        trial.set_user_attr(
            "shap_values", sum_of_all_shap_values[feature_idx.nonzero()]
        )
        trial.set_user_attr("selected_features", selected_features)
        # r2 = r2_score(true_y, predicted_y_proba)
        return calculate_adjusted_r2(true_y, predicted_y, number_of_coefficients)

    # try:
    #     optuna.study.delete_study(f"{target_feature_name}_iteration_{outer_cv_loop}", storage = "sqlite:///optuna_db.db")
    # except:
    #     print("new study")

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name="standard_lasso",
        direction="maximize",
        sampler=TPESampler(
            multivariate=False,
            n_startup_trials=3,
            consider_magic_clip=True,
            constant_liar=True,
        ),
    )
    # optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        # n_trials = 40,
        n_trials=20,
        n_jobs=1,
    )

    return dict(
        zip(
            study.best_trial.user_attrs["selected_features"],
            study.best_trial.user_attrs["shap_values"],
        )
    )
