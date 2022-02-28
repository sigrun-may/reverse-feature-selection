from typing import List, Tuple

import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from preprocessing import yeo_johnson_transform_test_train_splits
from src.reverse_feature_selection import (
    standard_lasso_feature_selection,
    reverse_lasso_feature_selection,
    rf_feature_selection,
)


def select_feature_subset(
    data: pd.DataFrame,
    remaining_data_indices,
    test_indices,
    outer_cv_loop_iteration: int,
    meta_data,
):
    print(
        "iteration: ",
        outer_cv_loop_iteration,
        " #############################################################",
    )

    preprocessed_data_dict = yeo_johnson_transform_test_train_splits(
        data_df=data.iloc[remaining_data_indices, :],
        outer_cv_loop_iteration=outer_cv_loop_iteration,
        meta_data=meta_data,
    )

    selected_feature_subset = {
        "standard_lasso": standard_lasso_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop_iteration, meta_data
        ),
        "reverse_lasso": reverse_lasso_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop_iteration, meta_data
        ),
        "random_forest": rf_feature_selection.select_features(
            preprocessed_data_dict, outer_cv_loop_iteration, meta_data
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

    return selected_feature_subset, test_indices, remaining_data_indices


# @Memory(settings.RESULT_DIRECTORY, verbose=5)
# def load_or_generate_feature_subsets(data_df):
#     return parallel_feature_subset_selection(data_df)


def load_or_generate_feature_subsets(data_df, meta_data):
    try:
        return joblib.load(meta_data["path_selected_subsets"])
    except:  # noqa
        return parallel_feature_subset_selection(data_df, meta_data)


def parallel_feature_subset_selection(data_df, meta_data) -> List[Tuple]:
    # outer cross-validation to validate selected feature subsets
    fold_splitter = StratifiedKFold(
        n_splits=meta_data["cv"]["n_outer_folds"], shuffle=True, random_state=42
    )
    # fold_splitter = LeaveOneOut()
    selected_subsets_and_indices = Parallel(
        n_jobs=meta_data["parallel"]["n_jobs_cv"], verbose=5
    )(
        delayed(select_feature_subset)(
            data_df, train_indices, test_indices, outer_cv_loop_iteration, meta_data
        )
        for outer_cv_loop_iteration, (train_indices, test_indices) in enumerate(
            fold_splitter.split(data_df.iloc[:, 1:], data_df["label"])
        )
    )
    # if settings.SAVE_RESULT:
    joblib.dump(
        selected_subsets_and_indices,
        meta_data["path_selected_subsets"],
        compress=("gzip", 3),
    )
    return selected_subsets_and_indices
