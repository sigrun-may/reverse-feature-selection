# Reverse feature selection for high-dimensioal data with tiny sample size [Sphinx-Doc](https://sigrun-may.github.io/reverse-feature-selection/)

## Overview

`reverse_feature_selection` is a Python package designed to help with feature selection in high-dimensional data with
tiny sample size. It provides tools to identify irrelevant features, improving stability of subsampling-based feature
selection methods. The package is designed to be used in combination with existing machine learning workflows.

 
[//]: # (## Features)

[//]: # ()
[//]: # (- Automated feature selection)

[//]: # (- Support for various machine learning models)

[//]: # (- Easy integration with existing workflows)

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (You can install the package using `pip`:)

[//]: # ()
[//]: # (```sh)

[//]: # (pip install reverse_feature_selection)

[//]: # (```)

## Usage

Here is a [basic example](reverse_feature_selection/basic_example.py) of how to use the `select_feature_subset` function. The function takes a pandas DataFrame as input
and returns a DataFrame with the selected features. 

```python
# variables list_of_random_seeds, data_pandas_dataframe, array_of_train_indices must be defined additionally
from reverse_feature_selection.reverse_rf_random import select_feature_subset

# Example metadata
meta_data = {
    'n_cpus': 4,
    'random_seeds': list_of_random_seeds,
    # train correlation threshold defines the features correlated to the target to be removed from the training data
    'train_correlation_threshold': 0.3,
}

# Select feature subset
result_df = select_feature_subset(data_pandas_dataframe, array_of_train_indices, meta_data)

# print names of non-zero features
print(f"Selected features: {result_df[result_df['feature_subset_selection'] > 0]}")
```

### Input Format
 - The input should be a pandas DataFrame. The first column must be named label and contain the target variable. 
All remaining columns are considered input features.

- Training Indices train_indices are a list of row indices indicating which rows are to be used for training.

The metadata dictionary must include the following keys:

    n_cpus: Integer specifying the number of CPU cores to use.

    random_seeds: A list of integers used as random seeds for reproducibility.

    train_correlation_threshold: Float. Threshold to exclude features that are highly correlated with the current 
                                        target feature. Features with an absolute correlation higher than this 
                                        threshold will not be considered during training.


### Output

The function returns a DataFrame with the selected feature subset in the 'feature_subset_selection' column. A numeric 
column where a value greater than zero indicates that the feature was selected.

[//]: # (- The first column should contain the labels and should be named 'label'.)

[//]: # (- The remaining columns should contain the features.)

[//]: # (- The function also requires a list of indices for the training data and a dictionary with metadata.)

[//]: # (- The metadata should contain the number of CPUs to use, a list of random seeds, and a threshold for the correlation between the features and the target variable.)

[//]: # (- The function returns a DataFrame with the selected features.)

[//]: # (- The selected features are indicated by a value greater than zero in the 'feature_subset_selection' column.)

[//]: # (- The function uses a random forest model to select the features.)

[//]: # (- The function uses a leave-one-out cross-validation approach to select the features.)

[//]: # (- The function uses a random seed to ensure reproducibility.)

[//]: # (- The function uses a threshold to remove features that are highly correlated with the target variable.)

[//]: # (- The function uses a threshold to remove features that are highly correlated with other features.)



## License

This project is licensed under the MIT License. See the LICENSE file for more information.\
Contact
For any questions or issues, please open an issue on GitHub or contact me at s.may(at)ostfalia.de.
