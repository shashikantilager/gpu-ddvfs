import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def cat_features_to_numeric():
    data_directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"

    data = pd.read_csv(data_directory + "final_data.csv")

    independent_variables = open("../filtered_independent_variables.txt").read().splitlines()
    # independent_variables = open("filtered_top_20_features").read().splitlines()

    categorical_features = open("../excluded_categorical_columns.txt").read().splitlines()
    independent_variables += categorical_features
    ## Keep the application name for reference, when clustering we can append the app name to the cluster number which would be helpful in decoding
    # applications that beong to different clusters. However, app name is excluded while clustering.
    columns = independent_variables + ["Application_name"]

    X = pd.DataFrame(data, columns=columns)

    # X = pd.DataFrame(data, columns=independent_variables)

    # https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html
    #  convert categorical columns to integers
    # cat_dims = [X.columns.get_loc(i) for i in categorical_features[:-1]]
    for header in categorical_features:
        data[header] = data[header].astype('category').cat.codes
    data.to_csv("final_data_categorical_converted.csv")

    for header in categorical_features:
        X[header] = X[header].astype('category').cat.codes

    X.to_csv("feature_set_categorical_converted_with_appname.csv")


def get_average():
    X = pd.read_csv("feature_set_categorical_converted_with_appname.csv")
    apps = X['Application_name'].unique()
    averaged_X = pd.DataFrame()
    for app in apps:
        # df = X.drop(['Application_name'], axis= 1)
        df = X
        appAverage = df[df['Application_name'] == app].mean()
        averaged_X[app] = appAverage

    df2 = averaged_X.transpose()
    df2['Application_name'] = df2.index
    df2.to_csv("feature_set_avgd_ctg_converted_with_appname.csv", index=False)


# -----------get defaultclockaverage

# def get_average_default_clock():
# scaler = StandardScaler()

X = pd.read_csv("feature_set_categorical_converted_with_appname.csv")
##Tesla P100
mem_clock = 715
sm_clock = 1189
X = X[(X['mem_clock_x'] == mem_clock) & (X['sm_clock'] == sm_clock)]
apps = X['Application_name'].unique()
norm_avg = pd.DataFrame()
for app in apps:
    df = X[X['Application_name'] == app]
    df = df.drop(['Application_name'], axis=1)
    # df = X
    # appAverage = df[df['Application_name'] == app].mean()
    # averaged_X[app] = appAverage

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    # transform = scaler.fit_transform(X)
    data = df.mean()
    norm_avg[app] = data

    # print(df)
    # df2 = averaged_X.transpose()
df2 = norm_avg.transpose()
df2['Application_name'] = df2.index
df2.to_csv("feature_set_normalisedavg_default_clock_ctg_converted_with_appname.csv", index=False)

# get_average_default_clock()
# get_average()

# cat_fatures_to_numeric()
# get_average()
# df = pd.read_csv("final_data_categorical_converted.csv")

#
# df2 = df.drop(['Application_name'], axis=1)

# df3
