import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
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

    def __init__(self, data_df, meta_data):
        """
        Initialize the CrossValidator class with data and metadata.

        Parameters:
        data_df (DataFrame): The dataset for cross-validation.
        meta_data (dict): The metadata related to the dataset and experiment.
        """
        ...
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
        dict: A dictionary containing the results of the cross-validation.
        """
        # StratifiedKFold outer cross-validation
        k_fold = StratifiedKFold(n_splits=self.meta_data["cv"]["n_outer_folds"], shuffle=True, random_state=2005)
        for outer_cv_loop, (train_index, test_index) in enumerate(
            k_fold.split(self.data_df.iloc[:, 1:], self.data_df["label"])
        ):
            logging.info(f"outer_cv_loop {outer_cv_loop + 1} of {self.meta_data['cv']['n_outer_folds']}")

            # Preprocess the data and cache the results if not available yet
            preprocessing.preprocess_data(
                train_index, test_index, self.data_df, outer_cv_loop, self.meta_data, correlation_matrix=True
            )

            # Calculate raw values for calculating feature subsets
            self._train_to_calculate_feature_subsets(train_index, test_index, outer_cv_loop)

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

    def _train_to_calculate_feature_subsets(self, train_index, test_index, outer_cv_loop):
        """
        Calculate raw results to determine feature subsets using both
        reverse feature selection and a standard random forest.

        Parameters:
        train_index (array-like): Indices of training samples.
        test_index (array-like): Indices of testing samples.
        outer_cv_loop (int): The current loop iteration of the outer cross-validation.
        """
        # Calculate raw metrics for feature subset calculation with reverse feature selection
        result_df = reverse_rf.calculate_oob_errors_per_feature(
            data_df=self.data_df,
            meta_data=self.meta_data,
            outer_cv_loop=outer_cv_loop,
        )
        # Calculate feature subset with standard random forest feature importance
        feature_importance, shap_values, validation_metric, training_oob = standard_rf.optimize(
            train_index, test_index, self.data_df, self.meta_data
        )
        # Store the results
        result_df["standard"] = feature_importance
        result_df["shap_values"] = shap_values  # TODO remove SHAP values?
        self.standard_validation_metric_list.append(validation_metric)
        self.standard_training_metric_list.append(training_oob)
        self.cv_result_list.append(result_df)

        # Store the indices used in cross-validation for later subset validation
        self.cv_indices_list.append((train_index, test_index))
