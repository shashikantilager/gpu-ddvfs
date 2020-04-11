import pandas as pd


def filter_featureset():
    data = pd.read_csv("/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/final_data.csv")

    ## get totalfeatureset.txt
    independent_variables = open("totalfeatureset.txt").read().splitlines()
    feature_set = pd.DataFrame(data, columns=independent_variables)
    ## Exclude Cateogrical
    exclude_catgorical_columns = open("excluded_categorical_columns.txt").read().splitlines()
    feature_set = feature_set[feature_set.columns[~feature_set.columns.isin(exclude_catgorical_columns)]]

    ##delete all columns that have only ) value
    # Writtne the column values to the follwing file -- > filtered_independent_variables.txt
    feature_set = feature_set.loc[:, (feature_set != 0).any(axis=0)]  # # Shape- (3317, 117)

    filtered_columns = feature_set.columns
    filtered_columns = filtered_columns.to_series()
    filtered_columns.to_csv("filtered_independent_variables.txt", index=False)

    return filtered_columns
