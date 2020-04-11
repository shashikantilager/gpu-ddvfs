import joblib
# import warnings
# warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split

# # https://www.kaggle.com/mistrzuniu1/tutorial-eda-feature-selection-regression
# https://towardsdatascience.com/deep-dive-into-catboost-functionalities-for-model-interpretation-7cdef669aeed
# independent_variables = open("filtered_independent_variables.txt").read().splitlines()
# independent_variables = open("filtered_independent_variables.txt").read().splitlines()
#
# categorical_features = open("excluded_categorical_columns.txt").read().splitlines()
# independent_variables += categorical_features
#
# data_directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"
# datafile= data_directory + "final_data.csv"
# # data = pd.read_csv(data_directory + "final_data.csv")
#
# train = pd.read_csv(datafile, names = independent_variables)
# train = train.drop(columns = ['NaN'], axis= 1)
# print(train.shape)

independent_variables = open("./filtered_independent_variables.txt").read().splitlines()
# independent_variables = open("filtered_independent_variables.txt").read().splitlines()

categorical_features = open("./excluded_categorical_columns.txt").read().splitlines()
independent_variables += categorical_features
dependent_variable = 'pwr'

data_directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"
datafile = data_directory + "final_data.csv"
data = pd.read_csv(data_directory + "final_data.csv")

X = pd.DataFrame(data, columns=independent_variables)

# X=X.fillna(-1)

print(X.columns)

categorical_features_indices = [data.columns.get_loc(c) for c in categorical_features]

y = data[dependent_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                      random_state=52)


def perform_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # model = CatBoostRegressor(
    #     random_seed=400,
    #     loss_function='RMSE',
    #     iterations=400,
    # )

    model = CatBoostRegressor()

    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_valid, y_valid),
        # verbose=False
    )

    print("RMSE on training data: " + model.score(X_train, y_train).astype(str))
    print("RMSE on test data: " + model.score(X_test, y_test).astype(str))

    joblib.dump(model, "catboostpwrwithcatg.pkl")
    return model


# model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test)

# model =   joblib.load("catboostmode.pkl")
# https://towardsdatascience.com/deep-dive-into-catboost-functionalities-for-model-interpretation-7cdef669aeed
# TODO USe SHAP and Loss funciton values and select features top 20
def analyse_feature_importance():
    model = joblib.load("catboostpwrwithcatg.pkl")

    feature_score = pd.DataFrame(
        list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features)))),
        columns=['Feature', 'Score'])
    feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort',
                                              na_position='last')

    feature_score.to_csv("catboostpwrwithcatg_fs.csv")

    plt.rcParams["figure.figsize"] = (12, 7)
    ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
    ax.set_title("Catboost Feature Importance Ranking", fontsize=14)
    ax.set_xlabel('')

    rects = ax.patches

    labels = feature_score['Score'].round(6)

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.35, label, ha='center', va='bottom')

    plt.show()


def top_n_bar_plot():
    N = 1

    ind = np.arange(N)  # the x locations for the groups
    width = 0.40  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 4))

    # df =  feature_score.head(10)
    feature_score = pd.read_csv("catboostpwrwithcatg_fs.csv")
    df = feature_score.head(10)

    features = pd.Series.tolist(df.Feature)
    ## get the score
    score_list = pd.Series.tolist(df.Score)
    color = 'gray'
    print(y)
    # for score in score_list:
    for idx, score in enumerate(score_list):
        ax.bar(idx, score, width)
        # ax.bar(idx, score, width,color = 'white', edgecolor = 'black', hatch = 'xx') ## for bw
    # add some text for labels, title and axes ticks
    ax.set_ylabel("Score (log scale)", fontsize=12)
    ax.set_xlabel("Features", fontsize=12)
    ax.set_yscale('log')

    # ax.set_xticks((1, 2, 3, 4))

    ax.set_xticklabels(features, fontsize=10)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        # use + or - to adjust the postion as shown below
        # ax.text(i.get_x() + 0.10, i.get_height() + 0.04, str(round((i.get_height()), 2)), fontsize=10,
        ax.text(i.get_x() + 0.10, i.get_height(), str(round((i.get_height()), 2)), fontsize=10,
                color='black')

    # ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('FIFO', 'Morpheus', 'BFD', 'FFD'),loc='upper right', shadow=True,fontsize='large')
    # plt.savefig("prediction_models/rmse_time_comparison.pdf")

    plt.show()


# top_n_bar_plot()
# perform_model(X_train, y_train, X_valid, y_valid, X_test, y_test)
# analyse_feature_importance()
top_n_bar_plot()
