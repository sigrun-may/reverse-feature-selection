import settings
from preprocessing import get_data
from feature_subset_selection import compute_feature_subsets
from validation import _extract_indices_and_results, evaluate_feature_selection


if __name__ == "__main__":
    # meta_data_dict = settings.get_meta_data()
    meta_data_dict = settings.get_old_meta_data("t_24")
    data_df = get_data(meta_data_dict)

    # # save meta data for parsed data
    # meta_data_dict["data"]["columns"] = data_df.columns.tolist()
    # settings.save_meta_data(meta_data_dict)

    # select feature subsets
    feature_selection_result = compute_feature_subsets(data_df, meta_data_dict)

    # (
    #     feature_selection_result_dict,
    #     robustness_dict,
    #     binary_selections_dict,
    #     test_train_indices_list,
    # ) = _extract_indices_and_results(feature_selection_result)

    # validate feature subsets
    evaluation_result_dict = evaluate_feature_selection(
        feature_selection_result,
        data_df,
        meta_data_dict,
    )
