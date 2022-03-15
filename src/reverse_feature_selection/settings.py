import os
import yaml
from sqlitedict import SqliteDict


def get_meta_data():
    # get next experiment id
    # d_ = delete
    # t_ = test
    # p_ = paper
    sqllite_mydict = SqliteDict("../../meta_data_db.sqlite", autocommit=True)
    experiment_ids_d = []
    experiment_ids_t = []
    experiment_ids_p = []
    for key in sqllite_mydict.keys():
        if key[2:] and key[2:].isdigit():
            if key.startswith("d"):
                experiment_ids_d.append(int(key[2:]))
            if key.startswith("t"):
                experiment_ids_t.append(int(key[2:]))
            if key.startswith("p"):
                experiment_ids_p.append(int(key[2:]))
    sqllite_mydict.close()

    if not experiment_ids_d:
        id_d = "d_0"
    else:
        id_d = f"d_{max(experiment_ids_d) + 1}"

    if not experiment_ids_t:
        id_t = "t_0"
    else:
        id_t = f"t_{max(experiment_ids_t) + 1}"

    if not experiment_ids_p:
        id_p = "p_0"
    else:
        id_p = f"p_{max(experiment_ids_p) + 1}"

    ########################################################
    # Settings
    ########################################################
    experiment_id = id_t
    print(experiment_id)
    data_name = "artificial1"
    # data_name = "colon"

    print(experiment_id)
    folder = dict(data="../../data", experiments="../../experiments")
    experiment_path = f"{folder['experiments']}/{experiment_id}"

    data = dict(
        input_data_path=f"{folder['data']}/{data_name}.csv",
        meta_data_path=f"{folder['data']}/{data_name}.pkl",
        clustered_data_path=f"{folder['data']}/{data_name}_clustered.csv",
        columns=None,
        excluded_features=[
            "bm_12",
            "bm_13",
            "bm_15",
            "bm_16",
            "bm_18",
            "bm_19",
            "bm_20",
            "bm_21",
            "bm_22",
            "bm_23",
            "bm_25",
            "bm_27",
        ],
        # excluded_features=[
        #     "bm_0",
        #     "bm_1",
        #     "bm_2",
        #     "bm_3",
        #     "bm_4",
        #     "bm_5",
        #     "bm_6",
        #     "bm_7",
        #     "bm_8",
        #     "bm_9",
        #     "bm_10",
        #     "bm_11",
        #     "bm_26",
        #     "bm_14",
        #     "bm_17",
        #     "bm_24",
        #     "bm_28",
        #     "bm_29",
        # ],
        # excluded_features=None,
        cluster_correlation_threshold=None,
        number_of_features=2000,
        number_of_samples=None,
        pos_label=0,
    )
    # no  trial completed: 15
    # Threshold: 30
    # no further improvement: 35

    # Keine Ergebnisse nach patience
    # Keine Verbesserung
    # Kein Ergebnis über dem Grenzwert nach patience

    selection_method = dict(
        rf=dict(
            trials=40,
            pruner_threshold=0.5,
            path_mlFlow=None,
            extra_trees=True,
        ),
        lasso=dict(trials=40, pruner=20),
        reverse_lasso=dict(
            trials=40,
            pruner_patience=None,
            pruner_threshold=0.1,
            correlation_threshold=0.3,
            remove_deselected=False,
        ),
    )
    meta_data = dict(
        experiment_id=experiment_id,
        experiment_path=experiment_path,
        commit="446b6a809f139842b698ad1cd5e63cd5ed8653b5",
        data=data,
        cv=dict(n_outer_folds=6, n_inner_folds=5),
        parallel=dict(n_jobs_preprocessing=1, n_jobs_cv=5, cluster=True),
        selection_method=selection_method,
        validation=dict(k_neighbors=11, knn_method="distance", threshold=0.1, factor=1),
        path_selected_subsets=f"{experiment_path}/selected_subsets.pkl.gz",
        path_validation=f"{experiment_path}/validation.pkl.gz",
        path_validation_result=f"{experiment_path}/validation.csv",
        path_preprocessed_data=None,
        # path_preprocessed_data=f
        # "{experiment_path}/preprocessed_data/preprocessed_data_dict",
        correlation_matrix_path=f"{folder['data']}/"
        f"{data_name}_correlation_matrix.pkl.gz",
        cluster_dict_path=f"{folder['data']}/" f"{data_name}_clusters.pkl.gz",
    )
    try:
        os.mkdir(folder["experiments"])
        print(f"Directory {folder['experiments']} created")
    except FileExistsError:
        print(f"Directory {folder['experiments']} already exists")
        pass

    try:
        os.mkdir(experiment_path)
        print(f"Directory {experiment_path} created")
    except FileExistsError:
        print(f"Directory {experiment_path} already exists")
        pass

    if meta_data["path_preprocessed_data"]:
        try:
            os.mkdir(f"{experiment_path}/preprocessed_data")
            print(f"Directory {experiment_path}/preprocessed_data created")
        except FileExistsError:
            print(f"Directory {experiment_path}/preprocessed_data already exists")
            pass

    save_meta_data(meta_data)
    return meta_data


def get_old_meta_data(experiment_id):
    experiment_path = f"../../experiments/{experiment_id}"
    with open(f"{experiment_path}/meta_data.yaml", "r") as file:
        meta_data = yaml.safe_load(file)
    return meta_data


def save_meta_data(meta_data):
    sqllite_mydict = SqliteDict("../../meta_data_db.sqlite", autocommit=True)
    sqllite_mydict[meta_data["experiment_id"]] = meta_data
    sqllite_mydict.close()

    # Write a YAML representation of meta_data_dict
    with open(f"{meta_data['experiment_path']}/meta_data.yaml", "w+") as fp:
        yaml.dump(meta_data, fp, allow_unicode=True, default_flow_style=False)


# N_JOBS_PREPROCESSING = 5  # number of parallel threads
# N_JOBS = 1
# N_FOLDS_INNER_CV = 5
# N_FOLDS_OUTER_CV = 6
# INPUT_DATA_PATH = "../../data/colon.csv"
# # INPUT_DATA_PATH = "../../data/huge_data.csv"
# # INPUT_DATA_PATH = "../../data/artificial1.csv"
# DIRECTORY_FOR_PICKLED_FILES = "../../pickled_files"
# NUMBER_OF_FEATURES = 2001
# # EXPERIMENT_NAME = "artificial_5000_10_fold"
# # EXPERIMENT_NAME = f"artificial1_{NUMBER_OF_FEATURES}
# # _{N_FOLDS_INNER_CV}_inner_folds_loo_outer_folds"
# # EXPERIMENT_NAME = f"artificial1_{NUMBER_OF_FEATURES}
# # _{N_FOLDS_INNER_CV}_inner_folds_{N_FOLDS_OUTER_CV}_outer_folds"
# # EXPERIMENT_NAME = (
# #     f"huge_data_{NUMBER_OF_FEATURES}"
# #     f"_{N_FOLDS_INNER_CV}_inner_folds_"
# #     f"{N_FOLDS_OUTER_CV}_outer_folds_corr_0_4_gain"
# # )
# # EXPERIMENT_NAME = (
# #     f"colon_{NUMBER_OF_FEATURES}"
# #     f"_{N_FOLDS_INNER_CV}_inner_folds_"
# #     f"{N_FOLDS_OUTER_CV}_outer_folds_corr_0_4_gain"
# # )
# EXPERIMENT_NAME = (
#     f"colon_clustered{NUMBER_OF_FEATURES}"
#     f"_{N_FOLDS_INNER_CV}_inner_folds_"
#     f"{N_FOLDS_OUTER_CV}_outer_folds_corr_0_2"
# )
# NUMBER_OF_TRIALS = 80
# PATIENCE_BEFORE_PRUNING_OF_STUDY = 10
# CORRELATION_THRESHOLD_CLUSTER = 0.8
# CORRELATION_THRESHOLD_REGRESSION = 0.2  # vorher 0.2
# N_NEIGHBORS = 5
# SAVE_RESULT = False
# RESULT_DIRECTORY = "../../results/"
# PATH_TO_RESULT = f"../../results/{EXPERIMENT_NAME}_threshold_{CORRELATION_THRESHOLD_REGRESSION}_{CORRELATION_THRESHOLD_CLUSTER}.pkl.gz"
# # PATH_TO_RESULT = f"/vol/projects/smay/develop/reverse_lasso_feature_selection/results/{EXPERIMENT_NAME}_threshold_{CORRELATION_THRESHOLD_REGRESSION}_{CORRELATION_THRESHOLD_CLUSTER}.pkl.gz"
#
# CLASSIFY_INTERSECT_ONLY = False
#
# # create directory for pickled preprocessing files for the experiment
# # path_to_pickled_files = f"{DIRECTORY_FOR_PICKLED_FILES}/{EXPERIMENT_NAME}"
# PICKLED_FILES_PATH = f"{DIRECTORY_FOR_PICKLED_FILES}/{EXPERIMENT_NAME}"
# # path_to_pickled_files = "N:/{}/{}".format(DIRECTORY_FOR_PICKLED_FILES, EXPERIMENT_NAME)
# # try:
# #     os.mkdir(PICKLED_FILES_PATH)
# #     print(f"Directory {PICKLED_FILES_PATH} created")
# # except FileExistsError:
# #     print(f"Directory {PICKLED_FILES_PATH} already exists")
# #     pass
