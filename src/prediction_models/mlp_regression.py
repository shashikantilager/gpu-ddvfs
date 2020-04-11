import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/final_data.csv")

dependent_variable = 'pwr'
# dependent_variable = 'time'
#

# independent_variables = ['sm',  'sm_clock']

independent_variables = open("totalfeatureset.txt").read().splitlines()
feature_set = pd.DataFrame(data, columns=independent_variables)

# replace special charecaters in feature set
feature_set[independent_variables] = feature_set[independent_variables].replace(
    {'%': '', ',': '', '/s': '', 'GB': '', 'MB': '', 'B': '', 'K': ''}, regex=True)

exclude_catgorical_columns = open("excluded_categorical_columns.txt").read().splitlines()
feature_set = feature_set[feature_set.columns[~feature_set.columns.isin(exclude_catgorical_columns)]]

##delete all columns that have only ) value
feature_set = feature_set.loc[:, (feature_set != 0).any(axis=0)]  # # Shape- (3317, 117)
# #Convert all columns to float
feature_set = feature_set.astype(float)
# get feature set and target prediction variable
target = data[dependent_variable]

# Split the test data and train data
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(feature_set, target, test_size=0.3)

training_time = prediction_time = mse = rmse = accuracy = count = r2_scorevalue = 0
r2score = []
# fit a model
model = MLPRegressor(hidden_layer_sizes=(5,),
                     activation='relu',
                     solver='adam',
                     learning_rate='adaptive',
                     max_iter=1000,
                     learning_rate_init=0.01,
                     alpha=0.01)

training_time_start = time.time()
# fit a model

model_fit = model.fit(X_train, y_train)

training_time_end = time.time()

training_time += training_time_end - training_time_start

prediction_time_start = time.time()

model_predictions = model.predict(X_test)

prediction_time_end = time.time()

prediction_time += prediction_time_end - prediction_time_start

print(model_predictions)

mse += mean_squared_error(y_test, model_predictions)
rmse += np.sqrt(mse)
print("MSE- ", mse)
print("RMSE- ", rmse)
accuracy += model.score(X_test, y_test)
print("Accuracy - ", accuracy)

r2_scorevalue += r2_score(y_test, model_predictions)
print("r2_score: ", r2_scorevalue)
r2score.append(accuracy)

# sm ams sm clock
# MSE-  0.3207869697672938
# RMSE-  0.5663805873856322
# Accuracy -  0.9821162040572954
# r2_score:  0.9821162040572955
