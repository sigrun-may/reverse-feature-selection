import pandas as pd

# from pandasgui import show
import settings
from preprocessing import get_data
from feature_subset_selection import compute_feature_subsets
from validation import evaluate_feature_selection


if __name__ == "__main__":
    meta_data_dict = settings.get_meta_data()
    # meta_data_dict = settings.get_old_meta_data("t_59")
    # meta_data_dict["validation"]["threshold"] = 0.1
    data_df = get_data(meta_data_dict)

    # # save meta data for parsed data
    meta_data_dict["data"]["columns"] = data_df.columns.tolist()
    # # meta_data_dict["validation"]["threshold"]=0.05
    # # meta_data_dict['path_validation_result']='validation_sum.csv'
    settings.save_meta_data(meta_data_dict)

    # select feature subsets
    feature_selection_result = compute_feature_subsets(data_df, meta_data_dict)

    # validate feature subsets
    evaluation_result_dict = evaluate_feature_selection(
        feature_selection_result,
        meta_data_dict,
    )
    # result output
    df = pd.DataFrame()
    for method, result in evaluation_result_dict.items():
        print(" ")
        print(f"#########################  {method}")
        for k, v in result.items():
            print(k, v)
            df.loc[k, method] = v
    df.to_csv(meta_data_dict["path_validation_result"])
