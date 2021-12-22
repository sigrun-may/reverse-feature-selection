import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut

import settings
from preprocessing import yeo_johnson_transform_test_train_splits
from src.reverse_feature_selection import standard_lasso_feature_selection, reverse_lasso_feature_selection, \
    rf_feature_selection


def select_feature_subset(
    data: pd.DataFrame, train_indices, test_indices, outer_cv_loop_iteration: int
):
    print(
        "iteration: ",
        outer_cv_loop_iteration,
        " #############################################################",
    )
    # load or generate transformed and standardized train test splits with respective train correlation matrix
    transformed_data_path_iteration = (
        f"{settings.DIRECTORY_FOR_PICKLED_FILES}/{settings.EXPERIMENT_NAME}/"
        f"transformed_test_train_splits_dict_{outer_cv_loop_iteration}.pkl"
    )
    preprocessed_data_dict = yeo_johnson_transform_test_train_splits(
        settings.N_FOLDS_INNER_CV, data.iloc[train_indices, :], None
    )

    selected_feature_subset = {
        "standard_lasso": standard_lasso_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop_iteration
        ),
        "reverse_lasso": reverse_lasso_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop_iteration
        ),
        "random_forest": rf_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop_iteration
        ),
    }
    # print(selected_features)
    # selected_features = select_features(preprocessed_data_dict)
    # selected_features = selection_method_rf(preprocessed_data_dict)

    # # append metrics to overall result for outer cross-validation
    # for selection_method, selected_features in metrics_dict.items():
    #     r = validation_metrics_dict.get(selection_method, [])
    #     r.append(selected_features)
    #     validation_metrics_dict[selection_method] = r

    return selected_feature_subset, test_indices, train_indices


# @Memory(settings.RESULT_DIRECTORY, verbose=5)
# def load_or_generate_feature_subsets(data_df):
#     return parallel_feature_subset_selection(data_df)


def load_or_generate_feature_subsets_old(data_df):
    try:
        return joblib.load(settings.PATH_TO_RESULT)
    except:
        return parallel_feature_subset_selection(data_df)


def parallel_feature_subset_selection(data_df):
    # outer cross-validation to validate selected feature subsets
    # fold_splitter = StratifiedKFold(n_splits=settings.N_FOLDS_OUTER_CV)
    fold_splitter = LeaveOneOut()
    selected_subsets = Parallel(n_jobs=settings.N_JOBS, verbose=5)(
        delayed(select_feature_subset)(
            data_df,
            train_indices,
            test_indices,
            outer_cv_loop_iteration,
        )
        for outer_cv_loop_iteration, (train_indices, test_indices) in enumerate(
            fold_splitter.split(data_df.iloc[:, 1:], data_df["label"])
        )
    )
    # if settings.SAVE_RESULT:
    joblib.dump(selected_subsets, settings.PATH_TO_RESULT, compress=("gzip", 3))
    return selected_subsets
