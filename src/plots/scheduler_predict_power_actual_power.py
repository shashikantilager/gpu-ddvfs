import configparser
import os

import matplotlib.pyplot as plt
import pandas as pd

# dir =  "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/schedule-data/"
# job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_30/"

job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_50_kmeans/"

dir = job_dir
dmon_data_file_qeaware = dir + "qe_aware_data_driven_dvfs_scheduler/" + "qeaware_nvidiasmidmon/"
dmon_data_file_default = dir + "default_application_clocks_scheduler/" + "qeaware_nvidiasmidmon/"
dmon_data_file_max = dir + "max_application_clock_scheduler/" + "qeaware_nvidiasmidmon/"

jobs_data_qeaware = dir + "qe_aware_data_driven_dvfs_scheduler" + "/jobs_data.csv"
jobs_data_default = dir + "default_application_clocks_scheduler" + "/jobs_data.csv"
jobs_data_max = dir + "max_application_clock_scheduler" + "/jobs_data.csv"

## preprocess all dmon filescd de
config = configparser.ConfigParser(delimiters=("="))
path = os.path.join("/home/ubuntu/phd_data/code/GPUETM/GPUETM/etc/configs/config.ini")
config.read(path)
application_que = config['SCHEDULER']['applications_que'].replace(' ', '').split(",")

jobs_data_qeaware = pd.read_csv(jobs_data_qeaware)
#
# dmon_max =  pd.read_csv(dmon_data_file_max)
jobs_data_max = pd.read_csv(jobs_data_max)
#
#
# dmon_default =  pd.read_csv(dmon_data_file_default)
jobs_data_default = pd.read_csv(jobs_data_default)

# --------------------------------- dealine plot
jobs_data_max.set_index('Unnamed: 0', inplace=True)
jobs_data_default.set_index('Unnamed: 0', inplace=True)
jobs_data_qeaware.set_index('Unnamed: 0', inplace=True)
max = jobs_data_max.transpose()

default = jobs_data_default.transpose()
qeaware = jobs_data_qeaware.transpose()

# deadline = max[4]
# systemdeadline = max['sys_time_deadline'].astype(float)

x = ['CORR', 'particlefilter_naive', 'COVAR', 'GEMM', 'ATAX', 'myocyte', 'SYR2K', 'particlefilter_float',
     'SYRK', '2MM', 'backprop', 'lavaMD']

h = 6
w = 4
fig, ax = plt.subplots(figsize=(h, w))

# xlen = 12
# x = np.arange(xlen)
# plt.plot(x, systemdeadline, marker='*', color='black', label="system")
# plt.plot(x, maxdeadlineachived, marker='*', color='magenta', label="max")
# plt.plot(x, defaultdeadlineachived, marker='*', color='red', label="default")
# plt.plot(x, qeawaredeadlineachived, marker='*', color='blue', label="d-dvfs")
# plt.legend()
# plt.show()
df = pd.read_csv("/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data/job_level_dmon_avg_kmeans_w_50_RD.csv")

# set height of bar
actualpwr = df['QEAware']

predicted_power = qeaware['predicted_power'].astype(float)
actual_pwr = actualpwr.astype(float)
plt.plot(x, predicted_power, '--', marker='*', color='red', label="Predicted power")
plt.plot(x, actual_pwr, '--', marker='+', color='blue', label="Actual consumed power")

plt.xlabel('Applications', fontsize=14)

plt.ylabel('Power (Watts)', fontsize=14)

ax.set_xticklabels(x, rotation=45, ha='right', fontsize=10)
plt.legend()

filename = "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data-schedule-energy/" + "predictpower" + "w_50_kmeans_small2.pdf"
# plt.savefig(filename, bbox_inches='tight'
plt.savefig(filename, bbox_inches='tight')

plt.show()
