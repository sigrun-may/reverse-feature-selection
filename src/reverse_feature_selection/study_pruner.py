import optuna

epsilon = 1/1000

def study_pruner(trial, epsilon, warm_up_steps, patience):

    # pruning of complete study
    if trial.number >= warm_up_steps:
        try:
            # check if any trial was completed and not pruned
            _ = trial.study.best_value
        except:
            trial.study.stop()
            raise optuna.TrialPruned()

        evaluation_metrics_of_completed_trials = []
        for _trial in trial.study.trials:
            if _trial.state == optuna.trial.TrialState.COMPLETE:
                evaluation_metrics_of_completed_trials.append(_trial.value)

        if len(evaluation_metrics_of_completed_trials) > patience:
            evaluation_metrics_within_patience = evaluation_metrics_of_completed_trials[-patience:]
            evaluation_metrics_before_patience = evaluation_metrics_of_completed_trials[:patience]

            best_value_within_patience = max(evaluation_metrics_within_patience)
            best_value_before_patience = max(evaluation_metrics_before_patience)

            # was best value of study before the patience?
            if best_value_before_patience + epsilon > best_value_within_patience:
                trial.study.stop()
                raise optuna.TrialPruned()


