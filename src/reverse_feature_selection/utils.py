import numpy as np
import pandas as pd


def get_well_separated_data(data_df):
    raw_data = data_df.values[:, 1:]
    print(raw_data.shape)
    print(raw_data.shape[1])
    class1 = raw_data[:15, :]
    print(class1)
    print(class1.shape)
    class2 = raw_data[15:, :]
    print(class2)
    print(class2.shape)

    for i in range(class2.shape[1]):
        if not (
            (np.min(class2[:, i]) < np.max(class1[:, i]))
            or (np.min(class1[:, i]) < np.max(class2[:, i]))
        ):
            print(data_df.columns[i])


data = pd.read_csv("../../data/artificial1.csv")
print(get_well_separated_data(data))
