import numpy as np
import pandas as pd


def get_well_separated_data(data_df):
    raw_data = data_df.values[:, 1:]
    print(raw_data.shape)
    class1 = raw_data[:15, :]
    print("class1.shape", class1.shape)
    class2 = raw_data[15:, :]
    print("class2.shape", class2.shape)

    for i in range(class2.shape[1]):
        if not ((np.min(class2[:, i]) < np.max(class1[:, i])) or (np.min(class1[:, i]) < np.max(class2[:, i]))):
            print(data_df.columns[i])


def sort_list_of_tuples_by_index(list_to_be_sorted, index=1, ascending=True):
    # getting length of list of tuples
    lst = len(list_to_be_sorted)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if (ascending and list_to_be_sorted[j][index] > list_to_be_sorted[j + 1][index]) or (
                not ascending and (list_to_be_sorted[j][index] < list_to_be_sorted[j + 1][index])
            ):
                temp = list_to_be_sorted[j]
                list_to_be_sorted[j] = list_to_be_sorted[j + 1]
                list_to_be_sorted[j + 1] = temp
    return list_to_be_sorted


def evaluate_proportion_of_selected_artificial_features(feature_set):
    random_features_list = []
    relevant_features_list = []
    pseudo_relevant_features_list = []
    for element in feature_set:
        if "random" in element:
            random_features_list.append(element)
        elif "bm" in element:
            relevant_features_list.append(element)
        else:
            pseudo_relevant_features_list.append(element)

    print("random_features", len(random_features_list), random_features_list)
    print("bm", len(relevant_features_list), relevant_features_list)
    print("pseudo", len(pseudo_relevant_features_list), pseudo_relevant_features_list)


def _select_intersect_from_feature_subsets(selected_feature_subsets, intersect):
    intersect_dict_list = []
    for subset_dict in selected_feature_subsets:
        intersect_dict = dict()
        for key, value in subset_dict.items():
            if key in intersect:
                intersect_dict[key] = value
        intersect_dict_list.append(intersect_dict)
    return intersect_dict_list
