# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


import pickle

import pandas as pd

from src.reverse_feature_selection.data_loader_tools import load_data_with_standardized_sample_size
import evaluate_feature_selection


def evaluate_reproducibility(input_data_df: pd.DataFrame, base_path: str, repeated_experiments: list[str]):
    results_dict = {}
    for experiment_id in repeated_experiments:
        subset_result_dict_final, performance_result_dict_final = evaluate_feature_selection.evaluate(
            input_data_df,
            f"{base_path}/{experiment_id}_result_dict.pkl",
            feature_selection_methods=[],
        )
        print(subset_result_dict_final)
        # compare the results


list_of_experiments = ["colon00", "colon01", "colon02", "colon03"]
pickle_base_path = f"../../results"
input_data_df = load_data_with_standardized_sample_size("colon")

evaluate_reproducibility(input_data_df, pickle_base_path, list_of_experiments)
