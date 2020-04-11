import matplotlib.pyplot as plt
import pandas as pd

data = "/home/shashi/phd_data/code/GPUETM-nectar/GPUETM/src/prediction_models/featureanalysis/catboost_threshold_analysis_loss_pwr2.csv"

df = pd.read_csv(data)
df.columns = ['key', 'index', 'accuracy', 'mse', 'rmse', 'training_time', 'prediction_time']

h = 8
w = 6
fig, ax = plt.subplots(figsize=(h, w))

# plt.plot(max_exec_list, marker='*', color='magenta', label="max")
# df = df.head(25)
plt.plot(df['index'], df['mse'], marker='x', color='red')
plt.xlabel("Number of sorted ranked features")
plt.ylabel("MSE")

# plt.savefig("/home/shashi/phd_data/code/GPUETM-nectar/GPUETM/src/prediction_models/featureanalysis/threshold_mse_time_loss_top25.pdf")
