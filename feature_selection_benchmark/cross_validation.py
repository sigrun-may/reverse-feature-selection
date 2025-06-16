# Copyright (c) 2024 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Cross-validation tools."""
from datetime import datetime, UTC
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
    meta_data["benchmark"] = {}

    # restrict process to specific CPU cores
    proc = psutil.Process()

    # logical cores 0 to 51 (26 physical cores with hyperthreading)
    all_cores = list(range(52))

    # exclude logical cores 0 and 26 (hyperthreads of the first physical core)
    used_cores = [core for core in all_cores if core not in (0, 26)]
    num_cores_used = len(used_cores)

    # apply CPU affinity restriction
    proc.cpu_affinity(used_cores)
    logger.info(f"Using logical cores (excluding 0 and 26): {used_cores}")
    logger.info(f"Number of parallel jobs defined in meta data: {meta_data['n_cpus']}")

    start_time_cv = datetime.now(tz=UTC)
    wall_times_list = []

    psutil.cpu_percent(interval=None, percpu=True)  # warm up CPU usage
    cpu_percentage_list = []
    cpu_util_percent_list = []
    cpu_time_per_core_list = []

    cv_result_list = []
    loo = LeaveOneOut()
    for fold_index, (train_indices, _) in enumerate(loo.split(data_df)):
        logger.info(f"fold_index {fold_index + 1} of {data_df.shape[0]}")
        cpu_times_before = proc.cpu_times()
        start_perf_counter = perf_counter()

        # Calculate raw values and feature subsets
        selected_feature_subset = feature_selection_function(
            data_df=data_df, train_indices=train_indices, meta_data=meta_data
        )
        wall_time = perf_counter() - start_perf_counter
        wall_times_list.append(wall_time)
        cpu_times_after = proc.cpu_times()
        cpu_percentage_list.append(psutil.cpu_percent(interval=None, percpu=True))

        # Calculate CPU time used (user + system time)
        user_time = cpu_times_after.user - cpu_times_before.user
        sys_time = cpu_times_after.system - cpu_times_before.system
        cpu_time_total = user_time + sys_time

        # Compute average CPU utilization during training (0-100%)
        cpu_util_percent_list.append((cpu_time_total / wall_time) * 100 if wall_time > 0 else 0)

        # Compute average CPU time per used virtual core
        cpu_time_per_core_list.append(cpu_time_total / num_cores_used if num_cores_used > 0 else 0)

        cv_result_list.append(selected_feature_subset)

    duration = datetime.now(tz=UTC) - start_time_cv
    logger.info(f"Duration of the cross-validation: {duration}")
    meta_data["benchmark"]["cv_duration"] = duration
    meta_data["benchmark"]["wall_times"] = wall_times_list
    meta_data["benchmark"]["cpu_usage"] = cpu_percentage_list
    meta_data["benchmark"]["average_cpu_util_percent"] = cpu_util_percent_list
    meta_data["benchmark"]["average_cpu_time_per_core"] = cpu_time_per_core_list
    return cv_result_list
