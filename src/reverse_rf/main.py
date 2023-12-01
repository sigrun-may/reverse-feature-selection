import numpy as np
import pickle

import datetime

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import reverse_rf
import preprocessing
import data_loader
from src.reverse_rf import standard_rf
import toml


# parse settings from toml
with open("../../settings.toml", "r") as file:
    meta_data = toml.load(file)


# data_df = pd.read_csv(
#     "/home/sigrun/PycharmProjects/ensemble-feature-selection-benchmark/data/overlapping/overlapping.csv"
# ).iloc[:, :500]
# assert data_df.columns[0] == 'label'

data_df = pd.read_csv("/home/sma19/PycharmProjects/reverse_feature_selection/data/small_50.csv")
assert data_df.columns[0] == "label"


# joblib.dump(data_df.columns, f"{data_name}_feature_names.pkl")

# # shorten artificial data for faster testing
# data_01 = data_df.iloc[:, 0:10]
# data_02 = data_df.iloc[:, 200:800]
# data_df = pd.concat([data_01, data_02], join="outer", axis=1)

# # data loaders
# X, y = data_loader.standardize_sample_size(*data_loader.load_colon_data())
# data_df = pd.DataFrame(X)
# data_df.insert(loc=0, column="label", value=y)
# # data_df.columns = data_df.columns.astype(str)
# data_df = data_df.reset_index(drop=True)
# assert data_df.columns[0] == "label"

print(data_df.shape)

start_time = datetime.datetime.utcnow()

cv_result_list = []
cv_indices_list = []
standard_validation_score_list = []

# outer cross-validation
# k_fold = StratifiedKFold(
#     n_splits=meta_data["cv"]["n_outer_folds"], shuffle=True, random_state=123098
# )
k_fold = StratifiedKFold(n_splits=meta_data["cv"]["n_outer_folds"], shuffle=True, random_state=2005)
for outer_cv_loop, (train_index, test_index) in enumerate(k_fold.split(data_df.iloc[:, 1:], data_df["label"])):
    print(f"outer_cv_loop {outer_cv_loop} of {meta_data['cv']['n_outer_folds']}")

    # preprocess data for reverse feature selection
    train_df, validation_df, corr_matrix_df = preprocessing.preprocess_data(
        train_index, test_index, data_df, correlation_matrix=True
    )
    # serialize the preprocessed data
    with open(
        f"../../preprocessed_data/{meta_data['data']['name']}_preprocessed_cv_fold_outer{outer_cv_loop}_train.pkl", "wb"
    ) as file:
        pickle.dump(train_df, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"../../preprocessed_data/{meta_data['data']['name']}_preprocessed_cv_fold_outer{outer_cv_loop}_corr.pkl", "wb"
    ) as file:
        pickle.dump(corr_matrix_df, file, protocol=pickle.HIGHEST_PROTOCOL)

    # # inner cross-validation
    # k_fold = LeaveOneOut()
    # for inner_cv_loop, (inner_train_index, inner_test_index) in enumerate(k_fold.split(data_df.iloc[train_index, :])):
    #     train_df, validation_df, corr_matrix_df = preprocessing.preprocess_data(
    #         inner_train_index, inner_test_index, data_df, correlation_matrix=True
    #     )
    #     # serialize the preprocessed data
    #     with open(f"data/{data_name}_preprocessed_cv_fold_outer{outer_cv_loop}_inner{inner_cv_loop}_train.pkl", "wb") as file:
    #         pickle.dump(train_df, file, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(f"data/{data_name}_preprocessed_cv_fold_outer{outer_cv_loop}_inner{inner_cv_loop}_validation.pkl", "wb") as file:
    #         pickle.dump(validation_df, file, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(f"data/{data_name}_preprocessed_cv_fold_outer{outer_cv_loop}_inner{inner_cv_loop}_corr.pkl", "wb") as file:
    #         pickle.dump(corr_matrix_df, file, protocol=pickle.HIGHEST_PROTOCOL)

    # calculate feature subset with reverse feature selection
    result_df = reverse_rf.calculate_validation_metric_per_feature(
        data_df=data_df,
        meta_data=meta_data,
        outer_cv_loop=outer_cv_loop,
    )
    # calculate feature subset with standard random forest
    feature_importance, validation_score = standard_rf.optimize(train_index, test_index, data_df, meta_data)
    result_df["standard"] = feature_importance
    standard_validation_score_list.append(validation_score)

    cv_result_list.append(result_df)
    cv_indices_list.append((train_index, test_index))

print("standard_validation_score", np.mean(standard_validation_score_list))
end_time = datetime.datetime.utcnow()
print(end_time - start_time)

method_result_dict = {"rf": cv_result_list, "rf_standard_validation_score": standard_validation_score_list}
result_dict = {
    "meta_data": meta_data,
    "method_result_dict": method_result_dict,
    "indices": cv_indices_list,
}
joblib.dump(result_dict, f"../../results/{meta_data['experiment_id']}_result_dict.pkl")
