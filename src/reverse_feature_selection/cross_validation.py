import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

import preprocessing
import reverse_rf_random as reverse_rf
import standard_rf

# Set up logging
logging.basicConfig(level=logging.INFO)


class CrossValidator:
    """
    This class is used to perform an outer cross-validation on a given dataset.
    It calculates the raw values to determine feature subsets using both
    reverse feature selection and a standard random forest.
    """

    def __init__(self, data_df: pd.DataFrame, meta_data: dict):
        """
        Initialize the CrossValidator class with data and metadata.

        Args:
            data_df: The dataset for cross-validation.
            meta_data: The metadata related to the dataset and experiment.
        """
        self.data_df = data_df
        self.meta_data = meta_data
        self.cv_result_list = []  # List to store cross-validation results
        self.cv_indices_list = []  # List to store indices used in cross-validation
        self.standard_validation_metric_list = []  # List to store standard RF validation metrics
        self.standard_training_metric_list = []  # List to store standard RF training metrics

    def cross_validate(self) -> dict:
        """
        Perform outer cross-validation on the dataset.

        Returns:
            Results of the cross-validation.
        """
        # # StratifiedKFold outer cross-validation
        # k_fold = StratifiedKFold(n_splits=self.meta_data["cv"]["n_outer_folds"], shuffle=True, random_state=2005)
        # for fold_index, (train_indices, test_indices) in enumerate(
        #     k_fold.split(self.data_df.iloc[:, 1:], self.data_df["label"])
        # ):
        #     logging.info(f"fold_index {fold_index + 1} of {self.meta_data['cv']['n_outer_folds']}")
        loo = LeaveOneOut()
        for fold_index, (train_indices, test_indices) in enumerate(loo.split(self.data_df.iloc[:, 1:])):
            logging.info(f"fold_index {fold_index + 1} of {self.data_df.shape[0]}")

            # Preprocess the data and cache the results if not available yet
            preprocessing.preprocess_data(
                train_indices, test_indices, self.data_df, fold_index, self.meta_data, correlation_matrix=True
            )

            # Calculate raw values for calculating feature subsets
            self._train_to_calculate_feature_subsets(train_indices, test_indices, fold_index)

        logging.info(f"standard_validation_metric: {np.mean(self.standard_validation_metric_list)}")
        logging.info(f"standard_training_metric: {np.mean(self.standard_training_metric_list)}")

        cross_validation_result_dict = {
            "method_result_dict": {
                "rf_cv_result": self.cv_result_list,
                "rf_standard_validation_metric": self.standard_validation_metric_list,
                "rf_standard_training_metric": self.standard_training_metric_list,
            },
            "meta_data": self.meta_data,
            "indices": self.cv_indices_list,
        }
        return cross_validation_result_dict

    def _train_to_calculate_feature_subsets(self, train_indices: np.ndarray, test_indices: np.ndarray, fold_index: int):
        """
        Calculate raw results to determine feature subsets using both
        reverse feature selection and a standard random forest.

        Args:
            train_indices: Indices of training samples.
            test_indices: Indices of testing samples.
            fold_index: The current loop iteration of the outer cross-validation.
        """
        # Calculate raw metrics for feature subset calculation with reverse feature selection
        result_df = reverse_rf.calculate_oob_errors_for_each_feature(
            data_df=self.data_df,
            meta_data=self.meta_data,
            fold_index=fold_index,
        )
        # Calculate feature subset with standard random forest feature importance
        feature_importance, shap_values, validation_metric, training_oob = standard_rf.optimize(
            train_indices, test_indices, self.data_df, self.meta_data
        )
        # Store the results
        result_df["standard"] = feature_importance
        result_df["shap_values"] = shap_values  # TODO remove SHAP values?
        self.standard_validation_metric_list.append(validation_metric)
        self.standard_training_metric_list.append(training_oob)
        self.cv_result_list.append(result_df)

        # Store the indices used in cross-validation for later subset validation
        self.cv_indices_list.append((train_indices, test_indices))
