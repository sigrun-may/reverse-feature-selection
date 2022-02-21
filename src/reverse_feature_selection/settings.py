import os

########################################################
# Settings
########################################################
N_JOBS_PREPROCESSING = 5  # number of parallel threads
N_JOBS = 1
N_FOLDS_INNER_CV = 5
N_FOLDS_OUTER_CV = 6
INPUT_DATA_PATH = "../../data/colon.csv"
# INPUT_DATA_PATH = "../../data/huge_data.csv"
# INPUT_DATA_PATH = "../../data/artificial1.csv"
DIRECTORY_FOR_PICKLED_FILES = "../../pickled_files"
NUMBER_OF_FEATURES = 2001
# EXPERIMENT_NAME = "artificial_5000_10_fold"
# EXPERIMENT_NAME = f"artificial1_{NUMBER_OF_FEATURES}
# _{N_FOLDS_INNER_CV}_inner_folds_loo_outer_folds"
# EXPERIMENT_NAME = f"artificial1_{NUMBER_OF_FEATURES}
# _{N_FOLDS_INNER_CV}_inner_folds_{N_FOLDS_OUTER_CV}_outer_folds"
# EXPERIMENT_NAME = (
#     f"huge_data_{NUMBER_OF_FEATURES}"
#     f"_{N_FOLDS_INNER_CV}_inner_folds_"
#     f"{N_FOLDS_OUTER_CV}_outer_folds_corr_0_4_gain"
# )
# EXPERIMENT_NAME = (
#     f"colon_{NUMBER_OF_FEATURES}"
#     f"_{N_FOLDS_INNER_CV}_inner_folds_"
#     f"{N_FOLDS_OUTER_CV}_outer_folds_corr_0_4_gain"
# )
EXPERIMENT_NAME = (
    f"colon_clustered{NUMBER_OF_FEATURES}"
    f"_{N_FOLDS_INNER_CV}_inner_folds_"
    f"{N_FOLDS_OUTER_CV}_outer_folds_corr_0_2"
)
NUMBER_OF_TRIALS = 80
PATIENCE_BEFORE_PRUNING_OF_STUDY = 10
CORRELATION_THRESHOLD_CLUSTER = 0.8
CORRELATION_THRESHOLD_REGRESSION = 0.2  # vorher 0.2
N_NEIGHBORS = 5
SAVE_RESULT = False
RESULT_DIRECTORY = "../../results/"
PATH_TO_RESULT = f"../../results/{EXPERIMENT_NAME}_threshold_{CORRELATION_THRESHOLD_REGRESSION}_{CORRELATION_THRESHOLD_CLUSTER}.pkl.gz"
# PATH_TO_RESULT = f"/vol/projects/smay/develop/reverse_lasso_feature_selection/results/{EXPERIMENT_NAME}_threshold_{CORRELATION_THRESHOLD_REGRESSION}_{CORRELATION_THRESHOLD_CLUSTER}.pkl.gz"

CLASSIFY_INTERSECT_ONLY = False

# create directory for pickled preprocessing files for the experiment
# path_to_pickled_files = f"{DIRECTORY_FOR_PICKLED_FILES}/{EXPERIMENT_NAME}"
PICKLED_FILES_PATH = f"{DIRECTORY_FOR_PICKLED_FILES}/{EXPERIMENT_NAME}"
# path_to_pickled_files = "N:/{}/{}".format(DIRECTORY_FOR_PICKLED_FILES, EXPERIMENT_NAME)
try:
    os.mkdir(PICKLED_FILES_PATH)
    print(f"Directory {PICKLED_FILES_PATH} created")
except FileExistsError:
    print(f"Directory {PICKLED_FILES_PATH} already exists")
    pass
