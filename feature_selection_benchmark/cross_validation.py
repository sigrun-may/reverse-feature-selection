# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Cross-validation tools."""
import datetime
import logging
from time import perf_counter

import cpuinfo
import pandas as pd
import psutil
from sklearn.model_selection import LeaveOneOut

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cross_validate(data_df: pd.DataFrame, meta_data: dict, feature_selection_function) -> list[pd.DataFrame]:
    """Perform outer cross-validation and calculate raw values for determining feature subsets.

    Args:
        data_df: The dataset for feature selection.
        meta_data: The metadata related to the dataset and experiment.
        feature_selection_function: The function to use for feature selection.

    Returns:
        Results of the cross-validation.
    """
    # save hardware configuration
    meta_data["hardware"] = {
        "CPU cores (logical):": psutil.cpu_count(logical=True),
        "CPU cores (physical):": psutil.cpu_count(logical=False),
        "RAM total (GB):": psutil.virtual_memory().total / (1024**3),
        "CPU info": cpuinfo.get_cpu_info(),
    }
    # initialize benchmarks
    benchmark_id = f"benchmark_{meta_data['experiment_id']}"
    meta_data[benchmark_id] = {}

    start_time_cv = datetime.datetime.now(tz=datetime.timezone.utc)
    wall_times_list = []
    perf_counter_wall_times_list = []

    psutil.cpu_percent(interval=None)  # warm up CPU usage
    cpu_usage_list = []

    cv_result_list = []
    loo = LeaveOneOut()
    for fold_index, (train_indices, _) in enumerate(loo.split(data_df)):
        logger.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")
        start_datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        start_perf_counter = perf_counter()

        # Calculate raw values and feature subsets
        selected_feature_subset = feature_selection_function(
            data_df=data_df, train_indices=train_indices, meta_data=meta_data
        )

        perf_counter_wall_times_list.append(perf_counter() - start_perf_counter)
        wall_times_list.append(datetime.datetime.now(tz=datetime.timezone.utc) - start_datetime)
        cpu_usage_list.append(psutil.cpu_percent(interval=None))
        cv_result_list.append(selected_feature_subset)

    duration = datetime.datetime.now(tz=datetime.timezone.utc) - start_time_cv
    print("Duration of the cross-validation: ", meta_data["cv_duration"])

    meta_data[benchmark_id]["cv_duration"] = duration
    meta_data[benchmark_id]["wall_times_list"] = wall_times_list
    meta_data[benchmark_id]["perf_counter_wall_times_list"] = perf_counter_wall_times_list
    meta_data[benchmark_id]["cpu_usage_list"] = cpu_usage_list
    return cv_result_list
