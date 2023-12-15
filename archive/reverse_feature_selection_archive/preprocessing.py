from typing import Tuple, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler


def get_data(meta_data_dict) -> pd.DataFrame:
    data = pd.read_csv(meta_data_dict["data"]["input_data_path"])
    # data.insert(loc=1,
    #           column='perfect',
    #           value=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print("input data shape:", data.shape)

    # data_01 = data.iloc[:, 0:10]
    # data_02 = data.iloc[:, 200:1200]
    # data = pd.concat([data_01, data_02], join="outer", axis=1)

    # exclude features selected by the user
    if meta_data_dict["data"]["excluded_features"]:
        data = data.drop(labels=meta_data_dict["data"]["excluded_features"], axis=1)
        print(
            data.shape,
            f" excluded "
            f"{len(meta_data_dict['data']['excluded_features'])} "
            f"features: {meta_data_dict['data']['excluded_features']}",
        )

    # print(f"perfect features: '{utils.get_well_separated_data(data)}")
    # # # scores, min, max = filter_methods.get_scores(data.iloc[:, :100])
    # # # print(filter_methods.get_scores(data.iloc[:, :100]))
    # excl_features = (filter_methods.get_scores(data.iloc[:, :102]))
    # print(excl_features)
    # data = data.drop(labels=excl_features, axis=1)
    # print([x for x in data.columns if 'bm' in x])
    # meta_data_dict['data']['excluded_features'] = excl_features
    # import settings
    # settings.save_meta_data(meta_data_dict)
    # # print(utils.sort_list_of_tuples_by_index(scores, ascending = False))
    # # print(f"good features (max, min): "
    # #       f"'{filter_methods.get_scores(data.iloc[:, :100])}")

    # adapt the data shape
    if meta_data_dict["data"]["number_of_features"] is not None:
        if data.shape[1] > meta_data_dict["data"]["number_of_features"]:
            data = data.iloc[:, : meta_data_dict["data"]["number_of_features"]]
            assert len(data.columns) == meta_data_dict["data"]["number_of_features"]
        else:
            print(
                f"Data shape after clustering is {data.shape}. It is not "
                f"possible to select "
                f'{meta_data_dict["data"]["number_of_features"]} features.'
            )
    print("data shape:", data.shape)

    # cluster highly correlated features
    # TODO shuffle data
    if meta_data_dict["data"]["cluster_correlation_threshold"]:
        data, cluster_dict = cluster_data(data, meta_data_dict)
        print("clustered data shape", data.shape)

    return data


def preprocess_validation_train_splits(
    data_df: pd.DataFrame,
    outer_cv_loop_iteration: int,
    meta_data: Dict,
) -> Dict[str, List[Union[Tuple[np.array], str]]]:
    """Split and preprocess data.

    Split data to validation and train. Transform (Yeo Johnson) and scale each split.
    Calculate a pearson correlation matrix for each transformed and scaled train split.

    Args:
        data_df: original data
        outer_cv_loop_iteration: iteration of the outer cross-validation loop
        meta_data: metadata dict (see ... )

    Returns:
        Dict with transformed and scaled test/train sets with the respective pearson correlation matrix
        for each train set and the respective column names

    """

    k_fold = StratifiedKFold(n_splits=meta_data["cv"]["n_inner_folds"], shuffle=True, random_state=42)
    transformed_data_list = Parallel(n_jobs=meta_data["parallel"]["n_jobs_preprocessing"], verbose=5)(
        delayed(transform_and_preprocess_data)(train_index, validation_index, data_df)
        for sample_index, (train_index, validation_index) in enumerate(
            k_fold.split(data_df.iloc[:, 1:], data_df["label"])
        )
    )
    assert len(transformed_data_list) == meta_data["cv"]["n_inner_folds"]

    if meta_data["path_preprocessed_data"] is not None:
        joblib.dump(
            transformed_data_list,
            f"{meta_data['path_preprocessed_data']}_{outer_cv_loop_iteration}.pkl",
            compress="lz4",
        )

    # return {"transformed_data": transformed_data, "feature_names": data_df.columns}
    return transformed_data_list


def transform_and_preprocess_data(train_index, validation_index, data_df, correlation_matrix=True):
    """Scale and transform data and calculate train correlation matrix if specified.

    BoxCox transformation is applied if all values are positive otherwise Yeo-Johnson transformation.
    Correlation matrices are created based on Pearson.

    Args:
        train_index: Index for the train split.
        validation_index: Index for the validation split.
        data_df: Data to be transformed.
        correlation_matrix: If a correlation matrix for the train data should be generated.

    Returns:
        Tuple of validation_data, train_data and train_correlation_matrix.
    """
    assert not data_df.isnull().values.any(), "Missing values" + data_df.head()

    # remove label for transformation
    # labels must be in col 0 (first col)
    unlabeled_data = data_df.values[:, 1:]
    label = data_df.values[:, 0]

    # # workaround for https://github.com/scikit-learn/scikit-learn/issues/14959
    # scaler = StandardScaler(with_std=False)
    # scaled_train = scaler.fit_transform(unlabeled_data[train_indices])
    # scaled_test = scaler.transform(unlabeled_data[fold_index])
    #
    # # transform and standardize test and train data
    # power_transformer = PowerTransformer(
    #     copy=True, method="yeo-johnson", standardize=True
    # )
    # train = power_transformer.fit_transform(scaled_train)
    # test = power_transformer.transform(scaled_test)

    if np.min(unlabeled_data) > 0:
        # transform and standardize test and train data
        power_transformer = PowerTransformer(copy=True, method="box-cox", standardize=True)
        train = power_transformer.fit_transform(unlabeled_data[train_index])
        validation = power_transformer.transform(unlabeled_data[validation_index])
    else:
        # workaround for https://github.com/scikit-learn/scikit-learn/issues/14959
        scaler = StandardScaler(with_std=False)
        scaled_train = scaler.fit_transform(unlabeled_data[train_index])
        scaled_test = scaler.transform(unlabeled_data[validation_index])

        # transform and standardize test and train data
        power_transformer = PowerTransformer(copy=True, method="yeo-johnson", standardize=True)
        train = power_transformer.fit_transform(scaled_train)
        validation = power_transformer.transform(scaled_test)

    # # transform and standardize test and train data
    # power_transformer = PowerTransformer(
    #     copy=True, method="yeo-johnson", standardize=True
    # )
    # train = power_transformer.fit_transform(unlabeled_data[train_indices])
    # test = power_transformer.transform(unlabeled_data[fold_index])

    assert validation.shape == (len(validation_index), unlabeled_data.shape[1])
    assert train.shape == (len(train_index), unlabeled_data.shape[1])

    validation_pd = pd.DataFrame(validation, columns=data_df.columns[1:])
    assert not validation_pd.isnull().values.any()

    # calculate correlation matrix for train data
    train_pd = pd.DataFrame(train, columns=data_df.columns[1:])
    assert not train_pd.isnull().values.any()

    train_correlation_matrix = None
    if correlation_matrix:
        train_correlation_matrix = train_pd.corr(method="pearson")

    # add label to transformed data
    train_pd.insert(0, "label", label[train_index])
    validation_pd.insert(0, "label", label[validation_index])
    # train_data = np.column_stack((label[train_indices], train))
    # validation_data = np.column_stack((label[validation_indices], validation))
    return validation_pd, train_pd, train_correlation_matrix


def get_cluster_dict(correlation_matrix, meta_data):
    updated_correlation_matrix = correlation_matrix.copy()

    # find clusters and uncorrelated_features
    clusters_list = []
    target_feature_list = updated_correlation_matrix.columns
    for target_feature_name in target_feature_list:
        # Is the target feature already assigned to another cluster?
        if target_feature_name in updated_correlation_matrix.columns:
            correlated_feature_names = _get_correlated_features(
                target_feature_name,
                updated_correlation_matrix,
                meta_data["data"]["cluster_correlation_threshold"],
            )
            if len(correlated_feature_names) > 1:  # more than the initial feature?
                clusters_list.append(correlated_feature_names)

                # remove features already assigned to another cluster from updated correlation matrix
                updated_correlation_matrix.drop(labels=correlated_feature_names, inplace=True)
                updated_correlation_matrix.drop(labels=correlated_feature_names, axis=1, inplace=True)
                assert updated_correlation_matrix.shape[0] == updated_correlation_matrix.shape[1]

    # find cluster representatives:
    # the cluster member with the absolute highest correlation to all other cluster features
    clusters_dict = {}
    for cluster in clusters_list:
        cluster_representative = _calculate_cluster_representative(cluster, correlation_matrix)
        clusters_dict[cluster_representative] = cluster

    for k, v in clusters_dict.items():
        print(k, v)
    print(len(clusters_dict.keys()))

    clustered_data_dict = dict(clusters=clusters_dict)
    clustered_data_dict["uncorrelated_features"] = updated_correlation_matrix.columns

    if meta_data["cluster_dict_path"]:
        joblib.dump(clustered_data_dict, meta_data["cluster_dict_path"], compress="lz4")

    return clustered_data_dict


def cluster_data(data_df: pd.DataFrame, meta_data) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    ########################################################
    # Cluster correlated features
    ########################################################
    # load or generate correlation matrix
    if meta_data["correlation_matrix_path"]:
        try:
            correlation_matrix = joblib.load(meta_data["correlation_matrix_path"])
        except:  # noqa
            correlation_matrix = data_df.iloc[:, 1:].corr(method="spearman")
            joblib.dump(
                correlation_matrix,
                meta_data["correlation_matrix_path"],
                compress="lz4",
            )
            print(f'Calculated new correlation matrix and saved to {meta_data["correlation_matrix_path"]}')
    else:
        correlation_matrix = data_df.iloc[:, 1:].corr(method="spearman")
    assert data_df.shape[1] - 1 == correlation_matrix.shape[0] == correlation_matrix.shape[1]

    # load or calculate clusters
    if meta_data["cluster_dict_path"]:
        try:
            clustered_data_dict = joblib.load(meta_data["cluster_dict_path"])
            print(f'New clusters are reused from {meta_data["cluster_dict_path"]}')
        except:  # noqa
            print("New clusters are calculated")
            clustered_data_dict = get_cluster_dict(correlation_matrix, meta_data)
    else:
        clustered_data_dict = get_cluster_dict(correlation_matrix, meta_data)

    print(
        "number of features in clustered data: ",
        len(clustered_data_dict["clusters"].keys()) + len(clustered_data_dict["uncorrelated_features"]),
    )

    # generate clustered data_df
    clustered_data_df = data_df[clustered_data_dict["uncorrelated_features"]].copy()
    clustered_data_index = clustered_data_df.columns.union(clustered_data_dict["clusters"].keys())
    # append cluster representatives
    dict_of_cols = {}
    for key, values in clustered_data_dict["clusters"].items():
        dict_of_cols[f"cluster_{key}"] = data_df[key]
        clustered_data_index = clustered_data_index.union([key])
    clustered_data_df = pd.concat([pd.DataFrame(dict_of_cols), clustered_data_df], axis=1)
    clustered_data_df.insert(0, "label", data_df["label"])

    assert (
        clustered_data_df.shape[1]
        == len(clustered_data_dict["clusters"].keys())
        + len(clustered_data_dict["uncorrelated_features"])
        + 1  # the label
    ), (
        f"clustered_data_df.shape[1] {clustered_data_df.shape[1]}= (len(cluster_dict.keys()) - 1) "
        f"{(len(clustered_data_dict['clusters'].keys()))} +len(cluster_dict['uncorrelated_features']) "
        f"{len(clustered_data_dict['uncorrelated_features'])}+1"
    )
    print(clustered_data_df.shape)

    # save correlation matrix for clustered data
    eliminated_features = [item for item in data_df.columns[1:] if item not in clustered_data_index]
    correlation_matrix.drop(eliminated_features, inplace=True, axis=1)
    correlation_matrix.drop(eliminated_features, inplace=True, axis=0)
    assert correlation_matrix.shape[1] == clustered_data_df.shape[1] - 1  # exclude label
    if meta_data["clustered_correlation_matrix_path"]:
        joblib.dump(
            correlation_matrix,
            meta_data["clustered_correlation_matrix_path"],
            compress="lz4",
        )
    return clustered_data_df, clustered_data_dict


def _calculate_cluster_representative(cluster_feature_names, correlation_matrix_df):
    sum_of_correlation_coefficients_dict = {}
    # for every feature of the cluster calculate the correlations to each other feature of the cluster
    for base_feature in cluster_feature_names:
        # calculate sum of correlations from base_feature to all cluster features
        sum_of_correlation_coefficients_dict[base_feature] = np.sum(
            np.abs((correlation_matrix_df[base_feature][cluster_feature_names]))
        )
    # select the feature with the highest overall correlation to all other elements of the cluster
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
