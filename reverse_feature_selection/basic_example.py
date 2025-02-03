# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Basic example of how to use the reverse feature selection algorithm.

This is a basic example of how to use the reverse feature selection algorithm. The example generates a synthetic dataset
with 100 irrelevant features and two relevant features. The relevant features have an increased effect size compared to
the irrelevant features. The algorithm selects the relevant features and prints the names of the selected features.
"""
import numpy as np
import pandas as pd

from reverse_feature_selection.reverse_rf_random import select_feature_subset

rng = np.random.default_rng()

# Example data
n_samples = 30
n_irrelevant_features = 100
data_df = pd.DataFrame({f"feature{i+1}": rng.random(n_samples) for i in range(n_irrelevant_features)})

# Insert relevant features with increased effect size
n_relevant_features = 2
for i in range(n_relevant_features):
    regulated_class = rng.random(n_samples // 2) + (i + 1) * 2
    unregulated_class = rng.random(n_samples // 2) + (i + 1)
    # concatenate the two classes to one feature
    data_df[f"relevant_feature{i+1}"] = np.concatenate((regulated_class, unregulated_class))

# insert label in column zero

# construct the label with 15 samples of class 0 and 15 samples of class 1, the first 15 samples are class 0
label = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
data_df.insert(0, "label", label)

# Select indices of 29 samples simulating leave-one-out cross-validation
train_indices = rng.choice(data_df.index, size=29, replace=False)

# generate random seeds of type int with great variety
seeds = [29, 10, 17, 42, 213, 34, 1, 5, 19, 3, 23, 9, 7, 123, 234, 345, 456, 567, 678, 789, 890, 15, 333, 37, 45, 56]

# Example metadata
meta_data = {
    "n_cpus": 4,
    "random_seeds": seeds,
    # train correlation threshold defines the features correlated to the target to be removed from the training data
    "train_correlation_threshold": 0.7,
}

# Select feature subset
result_df = select_feature_subset(data_df, train_indices, meta_data)

# print names of non-zero features
print(f"Selected features: {result_df[result_df['feature_subset_selection'] > 0]}")
