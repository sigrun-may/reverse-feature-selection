import os

########################################################
# Settings
########################################################
N_JOBS_PREPROCESSING = 1  # number of parallel threads
N_JOBS = 1
N_FOLDS_INNER_CV = 10
# INPUT_DATA_PATH = "input_data/colon.csv"
# INPUT_DATA_PATH = 'input_data/leukemia.csv'
# INPUT_DATA_PATH = "input_data/lasso_data.csv"
INPUT_DATA_PATH = "../../data/small_50.csv"
DIRECTORY_FOR_PICKLED_FILES = "../../pickled_files"
NUMBER_OF_FEATURES = 50
# EXPERIMENT_NAME = "artificial_5000_10_fold"
EXPERIMENT_NAME = f"test_{NUMBER_OF_FEATURES}_{N_FOLDS_INNER_CV}_fold"
CORRELATION_THRESHOLD_CLUSTER = 0.85
CORRELATION_THRESHOLD_REGRESSION = 0.3
N_NEIGHBORS = 5
SAVE_RESULT = False
# PATH_TO_RESULT = f"results/{EXPERIMENT_NAME}_threshold_{CORRELATION_THRESHOLD_REGRESSION}_{CORRELATION_THRESHOLD_CLUSTER}.pkl.gz"
PATH_TO_RESULT = f"/vol/projects/smay/develop/reverse_lasso_feature_selection/results/{EXPERIMENT_NAME}_threshold_{CORRELATION_THRESHOLD_REGRESSION}_{CORRELATION_THRESHOLD_CLUSTER}.pkl.gz"

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
