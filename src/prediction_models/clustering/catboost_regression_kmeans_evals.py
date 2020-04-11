import configparser
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
### load the config files
from sklearn.preprocessing import MinMaxScaler

cwd = os.getcwd()
config = configparser.ConfigParser(delimiters=("="))
path = os.path.join(cwd, "etc/configs/config.ini")
config.read(path)

# application_que = ["2MM", "ATAX"]
# application_que = ["2MM"]
que = "applications_que = 2MM,  ATAX,  CORR,  COVAR,  GEMM,  particlefilter_float,  particlefilter_naive,  SYR2K,  SYRK,  lavaMD,  backprop,  myocyte"
application_que = que.replace(' ', '').split(",")

models_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/prediction_models/catboost/"


# def get_reference_application_from_kmeans():
def prdict_kmeans_cluster():
    kmeansfile = "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/clustering/Kmeans_cluster_5.pkl"
    kmeans = joblib.load(kmeansfile)
    #

    independent_variables = open("../filtered_independent_variables.txt").read().splitlines()

    categorical_features = open("../excluded_categorical_columns.txt").read().splitlines()
    independent_variables += categorical_features
    columns = independent_variables + ["Application_name"]

    featuresetdata = pd.read_csv(
        "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/clustering/feature_set_categorical_converted_with_appname.csv")

    application_que = ["SYRK"]
    for app in application_que:
        # default_clock_data= pd.read_csv("/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/default_clock_data.csv")
        print ("Proicessing APp:{}".format(app))
        mem_clock = 715
        sm_clock = 1189
        df = pd.DataFrame(featuresetdata, columns=columns)
        df = featuresetdata[(featuresetdata['mem_clock_x'] == mem_clock) & (featuresetdata['sm_clock'] == sm_clock)]
        default_clock_app_data = pd.DataFrame(df[df["Application_name"] == app])
        default_clock_app_data = default_clock_app_data.drop(['Application_name'], axis=1)

        scaler = MinMaxScaler()
        default_clock_app_data[default_clock_app_data.columns] = scaler.fit_transform(
            default_clock_app_data[default_clock_app_data.columns])
        # transform = scaler.fit_transform(X)
        default_clock_app_data_avg = df.mean()

        prediction_input = default_clock_app_data_avg.transpose()
        prediction_input = prediction_input.drop(prediction_input.index[0])

        clusterhead = kmeans.predict(prediction_input.to_numpy().reshape(1, -1))
        print(clusterhead)


# /home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/clustering/catboost_regression_kmeans_eval.py
# /home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/clustering/kmeans_cluster_label_data.csv
# ----write to csv
# df = pd.read_csv("/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/clustering/kmeans_cluster_label_data2.csv")
# clusterdata= df[['Application_name', 'kmeans_cluster_labels']]
# clusterdata.to_csv("apps_clusters.csv", index= False)


df = pd.read_csv("apps_selection_basedon_clusters.csv")

apps = df['Application_name']
app = df[df['Application_name'] == "backprop"]
print(app['corelated_app'])
average_mse = average_rmse = average_train_time = average_prediction_time = average_r2score = 0
model_data = dict()

models_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/prediction_models/catboost/"

data_directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"
apptotaldata = pd.read_csv(data_directory + "final_data.csv")
data = pd.read_csv(
    "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/clustering/feature_set_categorical_converted_with_appname.csv")
counter = 0

# dependentvariable = 'pwr'
dependentvariable = 'time'
for index, row in df.iterrows():

    print("App: {} corelated:{} ".format(row['Application_name'], row['corelated_app']))

    app_name = row['Application_name']
    corelated_app = row['corelated_app']

    app_data = data[(data["Application_name"] == app_name)]
    corelated_app_data = data[(data["Application_name"] == corelated_app)]

    ##The suffix chacater "K" is missing in the converted data for two apps. Hence add it back so it loads correct model
    if app_name == "SYR2" or app_name == "SYR":
        model_app_name = app_name + "K"
        modelname = models_dir + "/" + model_app_name + "_" + dependentvariable + ".pkl"
    else:
        modelname = models_dir + "/" + app_name + "_" + dependentvariable + ".pkl"

    model = joblib.load(modelname)

    mem_clock = corelated_app_data['mem_clock_x'].unique()
    mem_clock = mem_clock[0]
    sm_clock_list = corelated_app_data['sm_clock'].unique()

    for sm_clock in sm_clock_list:
        ## get input vector for the current clock set
        featureset = corelated_app_data[
            (corelated_app_data['mem_clock_x'] == mem_clock) & (corelated_app_data['sm_clock'] == sm_clock)]

        # print("############ {} ".format(prediction_input.shape))
        # # taking average of kernel values
        # # prediction_input = pd.DataFrame(prediction_input.mean()).transpose()
        # # prediction_input = pd.DataFrame(prediction_input.mean())
        # print("############ After {} ".format(prediction_input.shape))

        independent_variables = open(
            "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/filtered_independent_variables.txt").read().splitlines()
        categorical_features = open(
            "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/prediction_models/excluded_categorical_columns.txt").read().splitlines()
        independent_variables += categorical_features
        prediction_input = pd.DataFrame(featureset, columns=independent_variables)
        # prediction_input = prediction_input.mean()

        predicted_pwr = model.predict(prediction_input)
        print("{}: {}".format(predicted_pwr, predicted_pwr.mean()))
        prediction_mean = predicted_pwr.mean()

        appdatadf = apptotaldata[(apptotaldata["Application_name"] == app_name)]
        appdatadf = appdatadf[
            (appdatadf['mem_clock_x'] == mem_clock) & (appdatadf['sm_clock'] == sm_clock)]

        print("************** {}:".format(appdatadf.time.mean()))
        # ytest_mean = np.array(ytest).mean()
        ytest_mean = appdatadf.time.mean()
        y = [ytest_mean]
        # pred_y = np.array(prediction_mean)
        pred_y = [prediction_mean]

        mse = mean_squared_error(y, pred_y)
        rmse = np.sqrt(mse)
        print("RMSE: {}".format(rmse))
        average_mse += mse
        average_rmse += rmse
        # #
        counter += 1

        # predicted_time = time_model.predict(prediction_input)

    # X_test.fillna(X_test.mean())
    # # y_test.fillna(X_test.mean())
    # mse = mean_squared_error(y_test, model_predictions)
    # rmse = np.sqrt(mse)
    # average_mse += mse
    # average_rmse += rmse

    # ## Accuracy or r2_score
    # accuracy = model.score(X_test, y_test)
    # average_r2score += accuracy
    # print("Application: {} : R2_score: {}".format(application, accuracy))
    # model_data[application] = "{},{},{}".format(accuracy, mse, rmse)

print ("---------------------Avergae values---------------------------------------")
print(" MSE:{}, RMSE:{}".format(average_mse / counter, average_rmse / counter))
print("Average Values: MSE, RMSE")
print("({},{})".format(average_mse / counter, average_rmse / counter))
