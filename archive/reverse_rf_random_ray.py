# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


# import multiprocessing
# import numpy as np
# import pandas as pd
# # import ray
# from more_itertools import chunked
# import logging
# import preprocessing
# from reverse_rf_random import calculate_oob_errors
#
# logging.basicConfig(level=logging.INFO)
#
# # ray.init(ignore_reinit_error=True, num_cpus=multiprocessing.cpu_count())
#
#
# def calculate_oob_errors_for_each_feature(
#     data_df: pd.DataFrame, meta_data: dict, fold_index: int, train_indices: np.ndarray
# ):
#     """
#     Calculate the mean out-of-bag (OOB) errors for random forest regressors with different random seeds
#     for training data including the label and without the label for each feature.
#
#     Args:
#         data_df: The training data.
#         meta_data: The metadata related to the dataset and experiment.
#         fold_index: The current loop iteration of the outer cross-validation.
#         train_indices: Indices for the training split.
#
#     Returns:
#         A DataFrame containing lists of OOB scores for repeated analyzes of labeled and unlabeled data.
#     """
#
#     # Preprocess the data and cache the results if not available yet
#     corr_matrix_df, train_df = preprocessing.preprocess_data(
#         train_indices,
#         data_df,
#         fold_index,
#         meta_data,
#     )
#     scores_labeled_list = []
#     scores_unlabeled_list = []
#     for batch in chunked(data_df.columns[1:], multiprocessing.cpu_count()):
#         scores_labeled_chunk = []
#         scores_unlabeled_chunk = []
#
#         for target_feature_name in batch:
#             (
#                 labeled,
#                 unlabeled,
#             ) = _remote_calculate_validation_metrics_per_feature.remote(
#                 target_feature_name, train_df, corr_matrix_df, meta_data
#             )
#             scores_labeled_chunk.append(labeled)
#             del labeled
#             scores_unlabeled_chunk.append(unlabeled)
#             del unlabeled
#
#         # delete object ids from ray to free memory
#         loaded_unlabeled_validation_metrics = ray.get(scores_unlabeled_chunk)
#         del scores_unlabeled_chunk
#         scores_unlabeled_list.extend(loaded_unlabeled_validation_metrics)
#         del loaded_unlabeled_validation_metrics
#
#         loaded_labeled_validation_metrics = ray.get(scores_labeled_chunk)
#         del scores_labeled_chunk
#         scores_labeled_list.extend(loaded_labeled_validation_metrics)
#         del loaded_labeled_validation_metrics
#     assert len(scores_unlabeled_list) == len(scores_labeled_list) == len(data_df.columns[1:])
#     result_df = pd.DataFrame(
#         data=scores_unlabeled_list,
#         index=data_df.columns[1:],
#         columns=["unlabeled"],
#     )
#     result_df["labeled"] = scores_labeled_list
#     return result_df
#
#
# @ray.remote(num_returns=2)
# def _remote_calculate_validation_metrics_per_feature(target_feature_name, train_df, corr_matrix_df, meta_data):
#     return calculate_oob_errors(target_feature_name, train_df, corr_matrix_df, meta_data)
