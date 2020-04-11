import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

data_directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"

data = pd.read_csv(data_directory + "final_data.csv")

dependent_variable = 'pwr'

independent_variables = open("../filtered_independent_variables.txt").read().splitlines()
# independent_variables = open("filtered_top_20_features").read().splitlines()

categorical_features = open("../excluded_categorical_columns.txt").read().splitlines()
independent_variables += categorical_features

# X=X.fillna(-1)
X = pd.DataFrame(data, columns=independent_variables)
# print(X.columns)
y = data[dependent_variable]

#  convert categorical columns to integers
cat_dims = [X.columns.get_loc(i) for i in categorical_features[:-1]]
for header in categorical_features:
    X[header] = X[header].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = CatBoostRegressor()
# grid_parameters = {'depth': [3,1,2,6,4,5,7,8,9,10],
#               'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.13, 0.15,0,2],
#               'iterations': [30, 50, 100,200,400,600,800,100],
#               # 'loss_function': ['RMSE', 'MultiRMSE', 'MAE',  'Quantile', 'LogLinQuantile', 'Poisson'],
#               'l2_leaf_reg': [1, 3, 5, 7, 9, 10,50, 100],
#               # 'border_count':[32,5,10,20,50,100,200],
#               # 'ctr_border_count':[50,5,10,20,100,200],
#               }
# paramter old used
# grid_parameters = {'depth': [3,1,2,6,4,5,7,8,9,10],
#                     'learning_rate': [0.01,0.02,0.03,0.05,0.07, 0.1,0.15],
#               'iterations': [30, 50, 100,200,400,600,800,1000,1200],
#               'l2_leaf_reg': [1, 3, 5, 7, 9, 10,50, 100],
#                 # 'border_count':[32,5,10,20,50,100,200],
#               }
#
# parameters from local
grid_parameters = {'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
                   'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.13, 0.15, 0.2],
                   'iterations': [30, 50, 100, 200, 400, 600, 800, 100, 500, 800, 1000, 1200],
                   'l2_leaf_reg': [1, 3, 5, 7, 9, 10, 50, 100],
                   'border_count': [32, 5, 10, 20, 50, 100, 200, 300, 400]
                   }

##Used one
grid_search_result = model.grid_search(grid_parameters,
                                       X=X_train,
                                       y=y_train,
                                       verbose=True,
                                       plot=True)
#                                        )
joblib.dump(grid_search_result, "gridsearchresultcatboost_pwr_recent.pkl")

# best_params = joblib.load("gridsearchresultcatboost_pwr.pkl")
# Results from Grid Search
# print("\n========================================================")
# print(" Results from Grid Search ")
# print("========================================================")
#
# print("\n The best estimator across ALL searched params:\n",
#       grid.best_estimator_)
#
# print("\n The best score across ALL searched params:\n",
#       grid.best_score_)
#
# print("\n The best parameters across ALL searched params:\n",
#       grid.best_params_)
# #
# time
# {'depth': 4, 'l2_leaf_reg': 3, 'iterations': 1200, 'learning_rate': 0.03}
# power
# {'depth': 6, 'l2_leaf_reg': 1, 'iterations': 600, 'learning_rate': 0.1}

##powerlocal , consider local one as better
# {'border_count': 400,
#  'depth': 4,
#  'l2_leaf_reg': 5,
#  'iterations': 1200,
#  'learning_rate': 0.1}
#
