from ctypes import Union
from typing import Dict, List, Tuple, Any

from optuna.samplers import TPESampler
from optuna import TrialPruned
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import celer
import shap

import numpy as np
from sklearn.metrics import r2_score
import settings
import optuna_study_pruner


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


def select_features(
    preprocessed_data_dict,
    test_index: int,
    # preprocessed_data_dict: Dict[str, List[Union[Tuple[Any], str]]],
) -> Dict[str, float]:
    """Select feature subset for best value of regularization parameter alpha.

    Args:
        preprocessed_data_dict: yj +pearson train data
        test_index: test index of outer cross-validation loop

    Returns: selected features + weights
    """

    def optuna_objective(trial):
        """Optimize regularization parameter alpha for lasso regression."""

        # if optuna_study_pruner.study_patience_pruner(
        #     trial, epsilon=0.001, warm_up_steps=20, patience=5
        # ) or optuna_study_pruner.study_no_improvement_pruner(
        #     trial,
        #     epsilon=0.01,
        #     warm_up_steps=30,
        #     number_of_similar_best_values=5,
        #     threshold=0.1,
        # ):
        #     print("study stopped")
        #     trial.study.stop()
        #     raise TrialPruned()

        # # pruning of complete study
        # if trial.number >= settings.PATIENCE_BEFORE_PRUNING_OF_STUDY:
        #     try:
        #         # check if any trial was completed and not pruned
        #         _ = trial.study.best_value
        #     except:
        #         trial.study.stop()
        #
        # #  adapt TPE sampler to suggest new parameter, when suggested parameter is repeated
        # alphas_history = []
        # for _trial in study.trials:
        #     if _trial.user_attrs:
        #         alphas_history.append(_trial.user_attrs["alpha"])
        #
        # alpha = trial.suggest_discrete_uniform("alpha", -5.0, 5.0, 0.01)
        # alpha = trial.suggest_loguniform("alpha", 0.001, 5.0)
        # # print(alphas_history)
        # alpha = trial.suggest_int("alpha", 1, 101, log=True) / 100
        # if alpha in alphas_history:
        #     # print('alpha', alpha)
        #     alpha = trial.suggest_int("beta", 1, 101, log=True) / 100
        #     # print('alpha_beta', alpha)
        # if alpha in alphas_history:
        #     # print('alpha', alpha)
        #     alpha = trial.suggest_int("gamma", 1, 101, log=True) / 100
        #     # print('alpha_gamma', alpha)
        #
        # # prune study when alpha is repeated
        # if alpha in alphas_history:
        #     print("repeated alpha: ", alpha, "in trial ", trial.number)
        #     raise TrialPruned()
        #
        # # save calculated alpha
        # trial.set_user_attr("alpha", alpha)

        predicted_y = []
        true_y = []
        number_of_coefficients = []
        coefficients = []
        all_shap_values = []
        sum_of_all_shap_values = []

        feature_names = preprocessed_data_dict["feature_names"]
        transformed_data = preprocessed_data_dict["transformed_data"]

        # cross validation for the optimization of alpha
        for test, train, train_correlation_matrix_complete in transformed_data:
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
                #alpha=trial.suggest_loguniform("alpha", 0.01, 5.0),
                alpha=trial.suggest_discrete_uniform("alpha", 0.001, 1.0,
                                                     0.001),
                verbose=0,
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
            # TODO count selection for each feature per fold/ per fold and
            #   sample (Robustness)

            # TODO unterschied coeff zu stacked SHAP

            # predict y_test
            predicted_y_test = lasso.predict(x_test)
            # predicted_y.append(predicted_y_test[0])
            predicted_y.extend(predicted_y_test)

        feature_idx = np.sum(np.array(sum_of_all_shap_values), axis=0)
        feature_names = feature_names.drop(["label"])
        selected_features = feature_names[feature_idx.nonzero()]
        nonzero_shap_values = feature_idx[feature_idx.nonzero()]
        assert len(selected_features) == feature_idx.nonzero()[0].size
        # trial.set_user_attr("label_coefficients", coefficients)
        trial.set_user_attr("shap_values", nonzero_shap_values)
        trial.set_user_attr("selected_features", selected_features)
        # r2 = r2_score(true_y, predicted_y_proba)

        # assume n = number of samples , p = number of independent variables
        # adjusted_r2 = 1-(1-R2)*(n-1)/(n-p-1)
        sample_size = len(true_y)
        adjusted_r2 = 1 - (
            ((1 - r2_score(true_y, predicted_y)) * (sample_size - 1))
            / (sample_size - (np.median(number_of_coefficients)) - 1)
        )
        return r2_score(true_y, predicted_y)
        # return calculate_adjusted_r2(true_y, predicted_y, number_of_coefficients)

    # try:
    #     optuna.study.delete_study(f"{target_feature_name}_iteration_{test_indices}", storage = "sqlite:///optuna_db.db")
    # except:
    #     print("new study")

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"standard_lasso_iteration_{test_index}",
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=3,
        ),
    )
    # optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        n_trials=settings.NUMBER_OF_TRIALS,
    )

    return dict(
        zip(
            study.best_trial.user_attrs["selected_features"],
            study.best_trial.user_attrs["shap_values"],
        )
    )
