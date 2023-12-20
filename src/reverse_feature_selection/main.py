import datetime
import pickle
from pathlib import Path

import pandas as pd
import toml
import git


from src.reverse_feature_selection.cross_validation import CrossValidator

# parse settings from toml
with open("../../settings.toml", "r") as file:
    meta_data = toml.load(file)

# save git commit hash
git_repository = git.Repo(search_parent_directories=True)
commit_sha = git_repository.head.object.hexsha
meta_data["git_commit_hash"] = commit_sha


# load data
data_df = pd.read_csv(meta_data["data"]["path"]).iloc[:, :500]
assert data_df.columns[0] == "label"

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

# save results
result_base_path = Path(f"../../results/")
# Create directory for saving the results
result_base_path.mkdir(parents=True, exist_ok=True)
with open(f"{result_base_path}/{meta_data['experiment_id']}_result_dict.pkl", "wb") as file:
    pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
