from typing import Tuple, Dict, List, Union, Optional

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import settings


def parse_data(number_of_features: int, path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    # indices = [0]
    # random_numbers = range(70, 120, 1)
    # indices.extend(random_numbers)
    # print(indices)
    # data = data.iloc[:, indices]
    # data = data.drop(labels=['bm_2', 'bm_3', 'bm_7', 'bm_8', 'bm_9', 'bm_10', 'bm_12', 'bm_18',
    #    'bm_26'], axis = 1)
    data = data.iloc[:, :number_of_features]
    print(data.shape)
    return data


def yeo_johnson_transform_test_train_splits(
    data_df: pd.DataFrame,
    num_inner_folds: int = settings.N_FOLDS_INNER_CV,
    path: Optional[str] = settings.PICKLED_FILES_PATH,
) -> Dict[str, List[Union[Tuple[np.array], str]]]:
    """
    Do test and train splits and transform (Yeo Johnson) them.

    :param data_df: original data
    :param num_inner_folds: number of inner cross-validation folds
    :param path: path to directory to pickle transformed data
    :return: the transformed data in selected_feature_subset_list dict with transformed test/train set for each sample and the column names
    """

    # k_fold = KFold(n_splits=num_inner_folds, shuffle=True, random_state=42)
    k_fold = StratifiedKFold(n_splits=num_inner_folds, shuffle=True, random_state=42)
    transformed_data = Parallel(n_jobs=settings.N_JOBS_PREPROCESSING, verbose=5)(
        delayed(transform_train_test_set)(train_index, test_index, data_df)
        for sample_index, (train_index, test_index) in enumerate(
            k_fold.split(data_df.iloc[:, 1:], data_df["label"])
        )
    )
    assert len(transformed_data) == settings.N_FOLDS_INNER_CV

    if path is not None:
        joblib.dump(
            {"transformed_data": transformed_data, "feature_names": data_df.columns},
            path + "/preprocessed_data.pkl",
        )

    return {"transformed_data": transformed_data, "feature_names": data_df.columns}


def transform_train_test_set(train_index, test_index, data_df):
    assert not data_df.isnull().values.any(), "Missing values" + data_df.head()

    # remove label for transformation
    # labels must be in col 0 (first col)
    unlabeled_data = data_df.values[:, 1:]
    label = data_df.values[:, 0]

    # # workaround for https://github.com/scikit-learn/scikit-learn/issues/14959
    # scaler = StandardScaler(with_std=False)
    # scaled_train = scaler.fit_transform(unlabeled_data[train_index])
    # scaled_test = scaler.transform(unlabeled_data[test_index])
    #
    # # transform and standardize test and train data
    # power_transformer = PowerTransformer(
    #     copy=True, method="yeo-johnson", standardize=True
    # )
    # train = power_transformer.fit_transform(scaled_train)
    # test = power_transformer.transform(scaled_test)

    if np.min(unlabeled_data) > 0:
        # transform and standardize test and train data
        power_transformer = PowerTransformer(
            copy=True, method="box-cox", standardize=True
        )
        train = power_transformer.fit_transform(unlabeled_data[train_index])
        test = power_transformer.transform(unlabeled_data[test_index])
    else:
        # workaround for https://github.com/scikit-learn/scikit-learn/issues/14959
        scaler = StandardScaler(with_std=False)
        scaled_train = scaler.fit_transform(unlabeled_data[train_index])
        scaled_test = scaler.transform(unlabeled_data[test_index])

        # transform and standardize test and train data
        power_transformer = PowerTransformer(
            copy=True, method="yeo-johnson", standardize=True
        )
        train = power_transformer.fit_transform(scaled_train)
        test = power_transformer.transform(scaled_test)

    # # transform and standardize test and train data
    # power_transformer = PowerTransformer(
    #     copy=True, method="yeo-johnson", standardize=True
    # )
    # train = power_transformer.fit_transform(unlabeled_data[train_index])
    # test = power_transformer.transform(unlabeled_data[test_index])

    assert test.shape == (len(test_index), unlabeled_data.shape[1])
    assert train.shape == (len(train_index), unlabeled_data.shape[1])

    # calculate correlation matrix for train data
    train_pd = pd.DataFrame(train, columns=data_df.columns[1:])
    assert not train_pd.isnull().values.any()
    train_correlation_matrix = train_pd.corr(method="pearson")

    # add label to transformed data TODO: pandas
    train_data = np.column_stack((label[train_index], train))
    test_data = np.column_stack((label[test_index], test))
    return test_data, train_data, train_correlation_matrix


def get_cluster_dict(correlation_matrix, threshold, path):
    updated_correlation_matrix = correlation_matrix.copy()

    # find clusters and uncorrelated_features
    clusters_list = []
    target_feature_list = updated_correlation_matrix.columns
    for target_feature_name in target_feature_list:
        # Is the target feature already assigned to another cluster?
        if target_feature_name in updated_correlation_matrix.columns:
            correlated_feature_names = _get_correlated_features(
                target_feature_name, updated_correlation_matrix, threshold
            )
            if len(correlated_feature_names) > 1:  # more than the initial feature?
                clusters_list.append(correlated_feature_names)

                # TODO else append to correlated features

                # remove features already assigned to another cluster from updated correlation matrix
                updated_correlation_matrix.drop(
                    labels=correlated_feature_names, inplace=True
                )
                updated_correlation_matrix.drop(
                    labels=correlated_feature_names, axis=1, inplace=True
                )
                assert (
                    updated_correlation_matrix.shape[0]
                    == updated_correlation_matrix.shape[1]
                )

    # find cluster representatives:
    # the cluster member with the absolute highest correlation to all other cluster features
    clusters_dict = {}
    for cluster in clusters_list:
        cluster_representative = _calculate_cluster_representative(
            cluster, correlation_matrix
        )
        clusters_dict[cluster_representative] = cluster

    clusters_dict["uncorrelated_features"] = updated_correlation_matrix.columns
    joblib.dump(clusters_dict, path, compress=("gzip", 3))

    for k, v in clusters_dict.items():
        print(k, v)
    print(len(clusters_dict.keys()))
    return clusters_dict


def cluster_data(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    ########################################################
    # Cluster correlated features
    ########################################################
    # load or generate correlation matrix
    correlation_matrix_path = f"{settings.PICKLED_FILES_PATH}/correlation_matrix.pkl.gz"
    try:
        correlation_matrix = joblib.load(correlation_matrix_path)
        # joblib.load('klkhlh')
    except:
        unlabeled_data_df = data_df.iloc[:, 1:]
        correlation_matrix = unlabeled_data_df.corr(method="spearman")
        joblib.dump(correlation_matrix, correlation_matrix_path, compress=("gzip", 3))
    assert (
        correlation_matrix.shape[0]
        == correlation_matrix.shape[1]
        == data_df.shape[1] - 1
    )

    # load or calculate clusters
    cluster_dict_path = f"{settings.PICKLED_FILES_PATH}/clusters.pkl.gz"
    try:
        cluster_dict = joblib.load(cluster_dict_path)
        # joblib.load('klkhlh')
    except:
        cluster_dict = get_cluster_dict(
            correlation_matrix,
            settings.CORRELATION_THRESHOLD_CLUSTER,
            cluster_dict_path,
        )

    print(cluster_dict)
    print(len(cluster_dict.keys()))

    # generate clustered data_df
    clustered_data_df = data_df[cluster_dict["uncorrelated_features"]].copy()

    for key, values in cluster_dict.items():
        # TODO dict ggf anders speichern: cluster and uncorrelated features
        if "uncorrelated_features" not in key:
            clustered_data_df.insert(0, f"cluster_{key}", data_df[key])

    clustered_data_df.insert(0, "label", data_df["label"])

    assert (
        clustered_data_df.shape[1]
        == (len(cluster_dict.keys()) - 1)
        + len(cluster_dict["uncorrelated_features"])
        + 1  # the label
    )
    print(clustered_data_df.shape)
    return clustered_data_df, cluster_dict


def _calculate_cluster_representative(cluster_feature_names, correlation_matrix_df):
    sum_of_correlation_coefficients_dict = {}
    # for every feature of the cluster calculate the correlations to each other feature of the cluster
    for base_feature in cluster_feature_names:
        # calculate sum of correlations from base_feature to all cluster features
        sum_of_correlation_coefficients_dict[base_feature] = np.sum(
            np.abs((correlation_matrix_df[base_feature][cluster_feature_names]))
        )

    # The feature_selection_method parameter takes selected_feature_subset_list function, and for each entry in the iterable,
    # it'll find the one for which the feature_selection_method function returns the highest selected_feature_subsets.
    return max(
        sum_of_correlation_coefficients_dict,
        key=sum_of_correlation_coefficients_dict.get,
    )


def _get_correlated_features(target_feature_name, correlation_matrix, threshold):
    """Return the feature indices correlated to the given target feature as numpy array."""

    correlations_to_target_feature = correlation_matrix[target_feature_name]
    correlated_feature_names = []
    for index, correlation_coefficient in enumerate(correlations_to_target_feature):
        # select indices of correlated features to exclude them from the training data
        if abs(correlation_coefficient) > threshold:
            correlated_feature_names.append(correlation_matrix.columns[index])

    return correlated_feature_names
