import pickle
import datetime
import pandas as pd
import toml

from src.reverse_feature_selection.cross_validation import CrossValidator

# parse settings from toml
with open("../../settings.toml", "r") as file:
    meta_data = toml.load(file)

print(type(meta_data))

# load data
data_df = pd.read_csv(meta_data["data"]["path"]).iloc[:, 1:]
assert data_df.columns[0] == "label"
data_df.to_csv(meta_data["data"]["path"])

# # shorten artificial data for faster testing
# data_01 = data_df.iloc[:, 0:10]
# data_02 = data_df.iloc[:, 200:800]
# data_df = pd.concat([data_01, data_02], join="outer", axis=1)

# # data loaders
# import data_loader
# X, y = data_loader.standardize_sample_size(*data_loader.load_colon_data())
# data_df = pd.DataFrame(X)
# data_df.insert(loc=0, column="label", value=y)
# # data_df.columns = data_df.columns.astype(str)
# data_df = data_df.reset_index(drop=True)
# assert data_df.columns[0] == "label"

print(data_df.shape)

start_time = datetime.datetime.utcnow()

cross_validator = CrossValidator(data_df, meta_data)
result_dict = cross_validator.cross_validate()

end_time = datetime.datetime.utcnow()
print(end_time - start_time)

with open(f"../../results/{meta_data['experiment_id']}_result_dict.pkl", "wb") as file:
    pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
