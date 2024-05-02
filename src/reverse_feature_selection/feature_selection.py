# class of feature selection methods
#
import pandas as pd
import reverse_rf_random as reverse_rf
import standard_rf


class FeatureSelection:
    """
    This class is used to select feature subsets from a given dataset.
    """

    def __init__(self, data_df: pd.DataFrame, meta_data: dict):
        """
        Initialize the FeatureSelection class with data and metadata.

        Args:
            data_df: The dataset for feature selection.
            meta_data: The metadata related to the dataset and experiment.
        """
        self.data_df = data_df
        self.meta_data = meta_data

    def select_feature_subsets(self, train_indices, fold_index) -> pd.DataFrame:
        """
        Select the features.

        Returns:
            Raw training results for feature selection.
        """
        if self.meta_data["method"] == "reverse_random_forest":
            return self.reverse_feature_selection(train_indices, fold_index)

        elif self.meta_data["method"] == "standard_random_forest":
            return self.standard_random_forest(train_indices)

        else:
            raise ValueError("Invalid feature selection method.")

    def reverse_feature_selection(self, train_indices, fold_index):
        """
        Perform reverse feature selection on the dataset.
        """
        # Calculate raw metrics for feature subset calculation with reverse feature selection
        return reverse_rf.calculate_oob_errors_for_each_feature(
            data_df=self.data_df,
            meta_data=self.meta_data,
            fold_index=fold_index,
            train_indices=train_indices,
        )

    def standard_random_forest(self, train_indices):
        """
        Perform standard random forest on the dataset.
        """
        # Calculate feature subset with standard random forest feature importance
        feature_importance, shap_values, training_oob_score = standard_rf.optimize(
            train_indices, self.data_df, self.meta_data
        )
        # Store the results
        result_df = pd.DataFrame()
        result_df["standard_rf"] = feature_importance
        result_df["shap_values_rf"] = shap_values

        return result_df
