import settings
from preprocessing import get_data
from feature_subset_selection import load_or_generate_feature_subsets
from validation import _extract_indices_and_results, \
    evaluate_feature_selection_methods


if __name__ == "__main__":
    meta_data = settings.get_meta_data()
    data_df = get_data(meta_data)
    meta_data["data"]["columns"] = data_df.columns.tolist()
    settings.save_meta_data(meta_data)

    # select feature subsets
    feature_selection_result = load_or_generate_feature_subsets(data_df, meta_data)
    (
        feature_selection_result_dict,
        robustness_dict,
        binary_selections_dict,
        test_train_indices_list,
    ) = _extract_indices_and_results(feature_selection_result)

    # validate feature subsets
    metrics_per_method_dict = evaluate_feature_selection_methods(
        feature_selection_result_dict,
        robustness_dict,
        binary_selections_dict,
        test_train_indices_list,
        data_df,
        meta_data,
    )
