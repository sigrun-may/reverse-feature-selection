# -*- coding: utf-8 -*-
# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data loader tools."""
from typing import List, Literal, Tuple

import pandas as pd
from mltb2.data import load_colon, load_leukemia_big, load_prostate


def load_data_with_standardized_sample_size(
    data: Literal["colon", "prostate", "leukemia_big"] = "colon"
) -> pd.DataFrame:
    """Load data with standardized sample size.

    The data is loaded and parsed from the internet.
    See mltb2.data.

    Args:
        data: Name of the data to load. Possible options are "colon", "prostate", "leukemia_big".

    Returns:
        Dataframe containing labels and data.
    """
    # generate function from string
    load_data_function = globals()[f"load_{data}"]

    # standardize sample size to balanced data
    label, data = load_data_function()
    x, y = standardize_sample_size(data, label)
    data_df = pd.DataFrame(x)
    data_df.insert(loc=0, column="label", value=y)
    data_df.columns = data_df.columns.astype(str)

    # reset index for cross validation splits
    data_df = data_df.reset_index(drop=True)
    assert data_df.columns[0] == "label"
    return data_df


def standardize_sample_size(data, label) -> Tuple[pd.DataFrame, pd.Series]:
    """Reduce samples to 15 for each class.

    Returns:
        Tuple containing balanced data and corresponding labels.
    """
    # reduce samples to 15 for each class
    indices: List[int] = []
    for i in range(label.shape[0]):
        if label.iloc[i] == 0:
            indices.append(i)
        if len(indices) == 15:
            break

    for i in range(label.shape[0]):
        if label.iloc[i] == 1:
            indices.append(i)
        if len(indices) == 30:
            break

    data = data.iloc[indices]
    label = label.iloc[indices]
    assert data.shape[0] == 30
    assert len(label) == 30

    print(data.shape)
    print(label.shape)

    return data, label
