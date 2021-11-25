from typing import Dict, Tuple, List, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut

import settings
from preprocessing import cluster_data, yeo_johnson_transform_test_train_splits


def parse_data(number_of_features: int, path: str) -> pd.DataFrame:
    data_df = pd.read_csv(path)
    data_df = data_df.iloc[:, :number_of_features]
    print(data_df.shape)
    return data_df


def selection_method_lasso(
        preprocessed_data_dict: Dict[str, List[Union[Tuple[np.array], str]]]
) -> Dict[str, float]:
    pass


def validate_feature_subset(data_df: pd.DataFrame, selected_feature_subset: Dict[str, float], neighbors: int) -> Dict[str, float]:
    pass

def evaluate_selected_features(data_df: pd.DataFrame, selected_feature_subset: Dict[str, float], neighbors: int) -> Dict[str, float]:
    pass


if __name__ == "__main__":
    data_df = parse_data(settings.NUMBER_OF_FEATURES, settings.INPUT_DATA_PATH)
    clustered_data_df, cluster_dict = cluster_data(data_df)

    loo = LeaveOneOut()
    feature_subsets = []
    all_results = {}
    for index, (train_index, test_index) in enumerate(loo.split(clustered_data_df)):
        # load or generate transformed and standardized train test splits with respective train correlation matrix
        transformed_data_path_iteration = (
            f"{settings.DIRECTORY_FOR_PICKLED_FILES}/{settings.EXPERIMENT_NAME}/"
            f"transformed_test_train_splits_dict_{str(index)}.pkl"
        )
        preprocessed_data_dict = yeo_johnson_transform_test_train_splits(settings.N_FOLDS_INNER_CV,
                                                                         clustered_data_df.iloc[train_index, :],
                                                                         None)

        selected_feature_subset = selection_method_lasso(preprocessed_data_dict)
        # selected_feature_subset = selection_method_reverse_lasso()
        # selected_feature_subset = selection_method_rf()

        metrics_dict = validate_feature_subset(clustered_data_df.iloc[test_index, :], selected_feature_subset, neighbors=3)

        # append metrics to overall result for outer cross-validation
        for k, v in metrics_dict.items():
            r = all_results.get(k, [])
            r.append(v)
            all_results[k] = r

    all_results = {k: np.mean(v) for k, v in all_results.items()}

    # evaluate_selected_features