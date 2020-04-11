import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils import utility

## TODO - Get que from the configuration

model_name = "catboost"
data_directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"
data = pd.read_csv(data_directory + "final_data.csv")
# # Initialize CatBoostRegressor
# model = CatBoostRegressor(iterations=2,
#                             learning_rate=1,
#                             depth=2)
# model = CatBoostRegressor()
#  power
model = CatBoostRegressor(iterations=1200,
                          learning_rate=0.1,
                          depth=6,
                          l2_leaf_reg=5)

##time
# model = CatBoostRegressor(iterations=1200,
#                             learning_rate=0.03,
#                             depth=4,
#                             l2_leaf_reg=3)

# application_que =  "2MM,  ATAX,  CORR,  COVAR,  GEMM,  particlefilter_float,  particlefilter_naive,  SYR2K,  SYRK"
application_que = "2MM,ATAX,CORR,COVAR,GEMM,particlefilter_float,particlefilter_naive,lavaMD,backprop,myocyte,SYR2K,SYRK"
application_que = application_que.replace(' ', '').split(",")

enable_model_dump_to_pickle = False

model_data = dict()

## For each application create a model based on leave one out validation
average_mse = average_rmse = average_train_time = average_prediction_time = average_r2score = 0

for application in application_que:

    ## For each application model, apply leave one out validation. Exclude the corresponding application data from training
    data = data[~(data["Application_name"] == application)]

    dependent_variable = 'pwr'
    # dependent_variable = 'time'

    independent_variables = open("filtered_independent_variables.txt").read().splitlines()
    # independent_variables = open("filtered_top_20_features").read().splitlines()

    categorical_features = open("excluded_categorical_columns.txt").read().splitlines()
    independent_variables += categorical_features

    feature_set = pd.DataFrame(data, columns=independent_variables)
    # get feature set and target prediction variable
    target = data[dependent_variable]
    # Split the test data and train data
    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(feature_set, target, test_size=0.3)
    # fit a model
    training_time_start = time.time()
    # fit a model
    model_fit = model.fit(X_train, y_train, cat_features=categorical_features)
    # model_fit = model.fit(X_train, y_train)
    training_time_end = time.time()

    training_time = training_time_end - training_time_start
    average_train_time += training_time

    prediction_time_start = time.time()

    model_predictions = model.predict(X_test)

    prediction_time_end = time.time()

    prediction_time = prediction_time_end - prediction_time_start
    average_prediction_time += prediction_time

    print(model_predictions)

    # X_test.fillna(X_test.mean())
    # y_test.fillna(X_test.mean())
    mse = mean_squared_error(y_test, model_predictions)
    rmse = np.sqrt(mse)
    average_mse += mse
    average_rmse += rmse

    ## Accuracy or r2_score
    accuracy = model.score(X_test, y_test)
    average_r2score += accuracy
    print("Application: {} : R2_score: {}".format(application, accuracy))
    model_data[application] = "{},{},{}".format(accuracy, mse, rmse)

    # Dump the model as pickle format.
    if enable_model_dump_to_pickle:
        model_dir = data_directory + "prediction_models/" + model_name + "/"
        file_name = application + "_" + dependent_variable + ".pkl"
        utility.Utils().dumpmodel(model, model_dir, file_name)
len = application_que.__len__()

print ("Application Model Data:")
print(model_data)

print ("---------------------Avergae values---------------------------------------")
print("R2Score:{}, MSE:{}, RMSE:{}, Train Time:{}, Prediction Time:{}".format(average_r2score / len,
                                                                              average_mse / len, average_rmse / len,
                                                                              average_train_time / len,
                                                                              average_prediction_time / len))
print("Average Values: R2Score, MSE, RMSE, Train_time, Prediction_time")
print("({},{},{}, {},{})".format(average_r2score / len,
                                 average_mse / len, average_rmse / len, average_train_time / len,
                                 average_prediction_time / len))
