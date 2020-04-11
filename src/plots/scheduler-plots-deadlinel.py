import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dir =  "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/schedule-data/"
# job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_50/"
job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_50_kmeans/"
from sklearn.preprocessing import minmax_scale

# job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/alphadata/syslevel_w_50_alpha_2/"
# job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_50/"

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
## arrival and deadline are same for all three policies
arrrivaltime = max['arrival_time'].astype(float)
systemdeadline = max['sys_time_deadline'].astype(float)

maxdeadlineachived = max['arrival_time'].astype(float) + max['scheduled_exec_time'].astype(float)
defaultdeadlineachived = default['arrival_time'].astype(float) + default['scheduled_exec_time'].astype(float)
qeawaredeadlineachived = qeaware['arrival_time'].astype(float) + qeaware['scheduled_exec_time'].astype(float)


# plt.show()

def connectpoints(x, y, p1, p2, color):
    plt.plot([x, p1], [y, p2], color, linewidth=0.6)


# --------------------------arrival deadline plot
def plot_arrival_deadline():
    h = 10
    w = 4
    fig, ax = plt.subplots(figsize=(h, w))
    ##old order
    # y_name = ['SYRK', 'backprop', 'CORR', '2MM', 'particlefilter_float', 'lavaMD',
    #      'COVAR', 'GEMM', 'particlefilter_naive', 'SYR2K', 'ATAX', 'myocyte']

    y_name = ['CORR', 'particlefilter_naive', 'COVAR', 'GEMM', 'ATAX', 'myocyte', 'SYR2K', 'particlefilter_float',
              'SYRK', '2MM', 'backprop', 'lavaMD']

    ylen = y_name.__len__()
    y_pos = np.arange(ylen)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_name, fontsize='small')

    # draw arrival time
    x_start = np.array(arrrivaltime)
    x_start = np.asfarray(arrrivaltime, float)
    plt.scatter(x_start, y_pos, color='blue', marker='+', s=6, linewidths=3, label="Arrival")
    #

    # Draw deadline
    x_finsh = np.array(systemdeadline)
    x_finsh = np.asfarray(x_finsh, float)
    plt.scatter(x_finsh, y_pos, color='red', marker='o', s=6, linewidths=3, label="Deadline")
    # plt.scatter(x_finsh, y_pos, marker=2, s=10)
    #

    # tick_spacing = 1000
    # #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ## connect arrival and deadline
    for num in y_pos:
        connectpoints(x_start[num], num, x_finsh[num], num, 'k-')

    plt.ylabel('Applications', fontsize=14)
    plt.xlabel('Time (sec)', fontsize=14)
    plt.legend()

    filename = "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data-schedule-energy/" + "arrival_deadlineplot" + "w_50_kmeans.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_normalised_deadline():
    # ---------------------------------deadline plot conencted

    #

    y_name = ['CORR', 'particlefilter_naive', 'COVAR', 'GEMM', 'ATAX', 'myocyte', 'SYR2K', 'particlefilter_float',
              'SYRK', '2MM', 'backprop', 'lavaMD']

    arrrivaltime = max['arrival_time'].astype(float)
    systemdeadline = max['sys_time_deadline'].astype(float)
    maxdeadlineachived = max['arrival_time'].astype(float) + max['scheduled_exec_time'].astype(float)
    defaultdeadlineachived = default['arrival_time'].astype(float) + default['scheduled_exec_time'].astype(float)
    qeawaredeadlineachived = qeaware['arrival_time'].astype(float) + qeaware['scheduled_exec_time'].astype(float)

    norm_arrrivaltime = np.zeros(arrrivaltime.__len__())
    norm_systemdeadline = np.zeros(arrrivaltime.__len__())
    norm_maxdeadlineachived = np.zeros(arrrivaltime.__len__())
    norm_defaultdeadlineachived = np.zeros(arrrivaltime.__len__())
    norm_qeawaredeadlineachived = np.zeros(arrrivaltime.__len__())

    for index, _ in enumerate(arrrivaltime):
        app_values = (
            arrrivaltime[index], systemdeadline[index], maxdeadlineachived[index], defaultdeadlineachived[index],
            qeawaredeadlineachived[index])
        app_values = np.array(app_values)
        # print("Index")
        # normalised=  (app_values - np.min(app_values))/np.ptp(app_values)
        normalised = minmax_scale(app_values, feature_range=(0, 1))
        norm_arrrivaltime[index] = normalised[0]
        norm_systemdeadline[index] = normalised[1]
        norm_maxdeadlineachived[index] = normalised[2]
        norm_defaultdeadlineachived[index] = normalised[3]
        norm_qeawaredeadlineachived[index] = normalised[4]
        print ("Index:{} App:{} Deadline: {} : NDeadline:{} QDeadline:{} Qdeadlinenorm:{}".format(index, y_name[index],
                                                                                                  systemdeadline[index],
                                                                                                  norm_systemdeadline[
                                                                                                      index],
                                                                                                  qeawaredeadlineachived[
                                                                                                      index],
                                                                                                  norm_qeawaredeadlineachived[
                                                                                                      index]))

    h = 10
    w = 4
    fig, ax = plt.subplots(figsize=(h, w))

    ylen = y_name.__len__()
    y_pos = np.arange(ylen)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_name, fontsize='small')

    # Draw deadline
    x_start = np.array(norm_systemdeadline)
    x_start = np.asfarray(x_start, float)
    plt.scatter(x_start, y_pos, color='red', marker='o', s=6, linewidths=3, label="Deadline")

    x_start = np.array(norm_maxdeadlineachived)
    x_start = np.asfarray(norm_maxdeadlineachived, float)
    plt.scatter(x_start, y_pos, color='black', marker="d", s=6, linewidths=3, label="MC")

    x_start = np.array(norm_defaultdeadlineachived)
    x_start = np.asfarray(norm_defaultdeadlineachived, float)
    plt.scatter(x_start, y_pos, color='blue', marker='x', s=6, linewidths=3, label="DC")

    x_start = np.array(norm_qeawaredeadlineachived)
    x_start = np.asfarray(norm_qeawaredeadlineachived, float)
    plt.scatter(x_start, y_pos, color='green', marker='*', s=6, linewidths=3, label="D-DVFS")

    # plt.scatter(x_finsh, y_pos, marker=2, s=10)
    #

    for num in y_pos:
        norm_app_values = (norm_systemdeadline[num], norm_maxdeadlineachived[num], norm_defaultdeadlineachived[num],
                           norm_qeawaredeadlineachived[num])

        norm_app_values = np.array(norm_app_values)
        x_start = norm_app_values.min()
        x_finsh = norm_app_values.max()
        connectpoints(x_start, num, x_finsh, num, 'k-')

    plt.ylabel('Applications', fontsize=14)
    plt.xlabel('Normalised completion time', fontsize=14)
    plt.legend()

    filename = "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data-schedule-energy/" + "normalised_deadlineplot" + "w_50_kmeans.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


plot_arrival_deadline()
