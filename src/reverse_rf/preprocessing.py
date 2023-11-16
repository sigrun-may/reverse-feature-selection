import math

from typing import Tuple, Dict, List, Union, Optional

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed


def get_data(meta_data_dict) -> pd.DataFrame:
    data = pd.read_csv(meta_data_dict["data"]["input_data_path"])
    # data.insert(loc=1,
    #           column='perfect',
    #           value=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print("input data shape:", data.shape)

    # shorten artificial data for faster testing
    data_01 = data.iloc[:, 0:10]
    data_02 = data.iloc[:, 200:700]
    data = pd.concat([data_01, data_02], join="outer", axis=1)

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
    # data.drop(labels=excl_features, axis=1, inplace=True)
    # print([x for x in data.columns if 'bm' in x])
    # meta_data_dict['data']['excluded_features'] = excl_features
    # import settings
    # settings.save_meta_data(meta_data_dict)
    # # print(utils.sort_list_of_tuples_by_index(excl_features, ascending = False))
    # # print(f"good features (max, min): "
    # #       f"'{filter_methods.get_scores(data.iloc[:, :100])}")

    # adapt the data shape
    if meta_data_dict["data"]["number_of_features"] is not None:
        if data.shape[1] > meta_data_dict["data"]["number_of_features"]:
            data = data.iloc[:, : meta_data_dict["data"]["number_of_features"]]
            assert len(data.columns) == meta_data_dict["data"]["number_of_features"]
        else:
            print(
                f"Data shape is {data.shape}. It is not "
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
    meta_data: Dict,
) -> Dict[str, List[Union[Tuple[np.array], str]]]:
    """Split and preprocess data.

    Split data to validation and train. Calculate a spearman correlation matrix for each train split.

    Args:
        data_df: original data
        outer_cv_loop_iteration: iteration of the outer cross-validation loop
        meta_data: metadata dict (see ... )

    Returns:
        Dict with validation/train sets with the respective spearman correlation matrix
        for each train set and the respective column names

    """

    k_fold = StratifiedKFold(
        n_splits=meta_data["cv"]["n_inner_folds"], shuffle=True, random_state=42
    )
    # k_fold_data_list = Parallel(n_jobs=meta_data["parallel"]["n_jobs_preprocessing"], verbose=5)(
    #     delayed(preprocess_data)(train_index, validation_index, data_df)
    #     for sample_index, (train_index, validation_index) in enumerate(
    #         k_fold.split(data_df.iloc[:, 1:], data_df["label"])
    #     )
    # )
    k_fold_data_list = []
    for train_index, validation_index in k_fold.split(
        data_df.iloc[:, 1:], data_df["label"]
    ):
        k_fold_data_list.append(preprocess_data(train_index, validation_index, data_df))
    assert len(k_fold_data_list) == meta_data["cv"]["n_inner_folds"]

    # if meta_data["path_preprocessed_data"] is not None:
    #     joblib.dump(
    #         k_fold_data_list,
    #         f"{meta_data['path_preprocessed_data']}_{outer_cv_loop_iteration}.pkl",
    #         compress="lz4",
    #     )
    return k_fold_data_list


def preprocess_data(train_index, validation_index, data_df, correlation_matrix=False):
    """Calculate train spearman correlation matrix if specified.

    Args:
        train_index: Index for the train split.
        validation_index: Index for the validation split.
        data_df: Complete input data.
        correlation_matrix: If a correlation matrix for the train data should be generated.

    Returns:
        Tuple of validation_data, train_data and train_correlation_matrix.
    """
    assert not data_df.isnull().values.any(), "Missing values" + data_df.head()

    train_pd = data_df.iloc[train_index, :]
    validation_pd = data_df.iloc[validation_index, :]

    # # enlarge effect size of label for reverse selection
    # train_pd.loc[:, 'label'] *= 100
    # validation_pd.loc[:, 'label'] *= 100

    assert validation_pd.shape == (len(validation_index), data_df.shape[1])
    assert train_pd.shape == (len(train_index), data_df.shape[1])

    train_correlation_matrix = None
    if correlation_matrix:
        unlabeled_train_df = data_df.iloc[train_index, 1:]
        assert unlabeled_train_df.shape[0] == len(train_index)
        assert (
            unlabeled_train_df.shape[1] == len(data_df.columns) - 1
        )  # exclude the label
        train_correlation_matrix = unlabeled_train_df.corr(method="spearman")
        assert (
            train_correlation_matrix.shape[0]
            == train_correlation_matrix.shape[1]
            == unlabeled_train_df.shape[1]
        )
    return train_pd, validation_pd, train_correlation_matrix


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

    for k, v in clusters_dict.items():
        print(k, v)
    print(len(clusters_dict.keys()))

    clustered_data_dict = dict(clusters=clusters_dict)
    clustered_data_dict["uncorrelated_features"] = updated_correlation_matrix.columns

    if meta_data["cluster_dict_path"]:
        joblib.dump(clustered_data_dict, meta_data["cluster_dict_path"], compress="lz4")

    return clustered_data_dict


def cluster_data(
    data_df: pd.DataFrame, meta_data
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
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
            print(
                f'Calculated new correlation matrix and saved to {meta_data["correlation_matrix_path"]}'
            )
    else:
        correlation_matrix = data_df.iloc[:, 1:].corr(method="spearman")
    assert (
        data_df.shape[1] - 1
        == correlation_matrix.shape[0]
        == correlation_matrix.shape[1]
    )

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
        len(clustered_data_dict["clusters"].keys())
        + len(clustered_data_dict["uncorrelated_features"]),
    )

    # generate clustered data_df
    clustered_data_df = data_df[clustered_data_dict["uncorrelated_features"]].copy()
    clustered_data_index = clustered_data_df.columns.union(
        clustered_data_dict["clusters"].keys()
    )
    # append cluster representatives
    dict_of_cols = {}
    for key, values in clustered_data_dict["clusters"].items():
        dict_of_cols[f"cluster_{key}"] = data_df[key]
        clustered_data_index = clustered_data_index.union([key])
    clustered_data_df = pd.concat(
        [pd.DataFrame(dict_of_cols), clustered_data_df], axis=1
    )
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
    eliminated_features = [
        item for item in data_df.columns[1:] if item not in clustered_data_index
    ]
    correlation_matrix.drop(eliminated_features, inplace=True, axis=1)
    correlation_matrix.drop(eliminated_features, inplace=True, axis=0)
    assert (
        correlation_matrix.shape[1] == clustered_data_df.shape[1] - 1
    )  # exclude label
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


def get_uncorrelated_train_and_validation_data(
    data_split, target_feature, labeled, meta_data
):
    train_df, validation_df, train_correlation_matrix_df = data_split
    assert train_df.shape[0] > validation_df.shape[0]

    # remove correlations to target feature
    unlabeled_train_df = train_df.iloc[:, 1:]
    uncorrelated_features_mask = (
        train_correlation_matrix_df[target_feature]
        .abs()
        .le(
            meta_data["data"]["train_correlation_threshold"],
            axis="index",
            # For a correlation matrix filled only with the lower half,
            # the first elements up to the diagonal would have to be read
            # with axis="index" and the further elements after the diagonal
            # with axis="column".
        )
    )
    uncorrelated_features_index = unlabeled_train_df.loc[
        :, uncorrelated_features_mask
    ].columns
    assert "label" not in uncorrelated_features_index
    assert (
        train_correlation_matrix_df.loc[target_feature, uncorrelated_features_mask]
        .abs()
        .max()
        <= meta_data["data"]["train_correlation_threshold"]
    ), f"{meta_data['data']['train_correlation_threshold']}"

    # check if train data would keep at least one feature after removing label and target_feature
    # if not uncorrelated_features_index.size > 0:
    #     index_min = train_correlation_matrix_df[target_feature].abs().idxmin()
    #     uncorrelated_features_index = uncorrelated_features_index.insert(0, index_min)
    # else:
    #     assert (train_correlation_matrix_df.loc[target_feature, uncorrelated_features_mask].abs().max() <= meta_data["data"][
    #         "train_correlation_threshold"]), f"{meta_data['data']['train_correlation_threshold']}"
    # assert uncorrelated_features_index.size > 0

    if labeled:
        # insert label
        uncorrelated_features_index = uncorrelated_features_index.insert(0, "label")
        assert "label" in uncorrelated_features_index

    if math.isclose(uncorrelated_features_index.size, 0):
        return None

    # prepare train data
    x_train = train_df[uncorrelated_features_index]
    y_train = train_df[target_feature].values.reshape(-1, 1)
    assert x_train.shape[1] == uncorrelated_features_index.size
    assert y_train.shape[1] == 1

    # prepare validation data
    x_validation = validation_df[uncorrelated_features_index]
    y_validation = validation_df[target_feature].values.reshape(-1, 1)
    assert x_validation.shape[1] == uncorrelated_features_index.size
    assert y_validation.shape[1] == 1

    return x_train, y_train, x_validation, y_validation
