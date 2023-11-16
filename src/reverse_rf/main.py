import datetime

import joblib
import pandas as pd
from numpy import ravel
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from src.reverse_fs import settings
import reverse_rf
import preprocessing
import data_loader
from src.reverse_rf import standard_rf

data_df = pd.read_csv(
    "/home/sigrun/PycharmProjects/ensemble-feature-selection-benchmark/data/overlapping/overlapping.csv"
).iloc[:, :500]
# data_df = pd.read_csv("/home/sigrun/PycharmProjects/reverse_feature_selection/data/small_50.csv")
# # data_df = pd.read_csv("/home/sigrun/PycharmProjects/reverse_feature_selection/data/generated_data_paper_5000.csv")
# data_df = pd.read_csv("/home/sigrun/PycharmProjects/reverse_feature_selection/data/separated.csv").iloc[:, :1000]
#
# # # assert data_df.columns[0] == 'label'
# # # shorten artificial data for faster testing
# # data_01 = data_df.iloc[:, 0:10]
# # data_02 = data_df.iloc[:, 200:800]
# # data_df = pd.concat([data_01, data_02], join="outer", axis=1)
#
# print(data_df.shape)
# joblib.dump(data_df.columns, "1000_feature_names.pkl")

# X, y = data_loader.standardize_sample_size(*data_loader.load_colon_data())

# # Generate a binary classification dataset.
# X, y = make_classification(
#     n_samples=30,
#     n_features=500,
#     # n_clusters_per_class=1,
#     n_informative=25,
#     random_state=123,
#     shuffle=False,
#     # class_sep=3,
# )

# data_df = pd.DataFrame(X)
# data_df.insert(loc=0, column="label", value=y)
# data_df.columns = data_df.columns.astype(str)
#
# print(data_df.shape)
# joblib.dump(data_df.columns, "wide500_feature_names.pkl")

meta_data = meta_data_dict = settings.get_meta_data()

start_time = datetime.datetime.utcnow()
result_dict = {}

# outer cross-validation
k_fold = StratifiedKFold(
    n_splits=meta_data["cv"]["n_outer_folds"], shuffle=True, random_state=42
)
for outer_cv_loop, (remain_index, test_index) in enumerate(
    k_fold.split(data_df.iloc[:, 1:], data_df["label"])
):
    print(f"iteration {outer_cv_loop}")
    # calculate feature subset with reverse feature selection
    result_df = reverse_rf.calculate_validation_metric_per_feature(
        train_index=remain_index,
        validation_index=test_index,
        data_df=data_df,
        meta_data=meta_data,
    )
    # calculate feature subset with standard random forest
    result_df["standard_rf"] = standard_rf.optimize(remain_index, data_df, meta_data)

    # save results
    result_dict[f"result_iteration_{outer_cv_loop}"] = result_df
    result_dict[f"test_index_iteration_{outer_cv_loop}"] = test_index
    result_dict[f"train_index_iteration_{outer_cv_loop}"] = remain_index
    result_dict["meta_data"] = meta_data
    joblib.dump(result_dict, f"overlapping_500_result_dict.pkl")
end_time = datetime.datetime.utcnow()
print(end_time - start_time)
