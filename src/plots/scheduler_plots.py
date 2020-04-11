import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data_processing

# dir =  "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/schedule-data/"
sys_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/syslevel_w_50/"
job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_50/"

dir = sys_dir
dmon_data_file_qeaware = dir + "qe_aware_data_driven_dvfs_scheduler/" + "all_job_nvidiasmidmon" + "/qe_aware_data_driven_dvfs_scheduler_dmon_data"
dmon_data_file_default = dir + "default_application_clocks_scheduler/" + "all_job_nvidiasmidmon" + "/default_application_clocks_scheduler_dmon_data"
dmon_data_file_max = dir + "max_application_clock_scheduler/" + "all_job_nvidiasmidmon" + "/max_application_clock_scheduler_dmon_data"

jobs_data_qeaware = dir + "qe_aware_data_driven_dvfs_scheduler/" + "/jobs_data.csv"
jobs_data_default = dir + "default_application_clocks_scheduler/" + "/jobs_data.csv"
jobs_data_max = dir + "max_application_clock_scheduler/" + "/jobs_data.csv"


## preprocess all dmon files
def preprocess_dmon_data(dmon_data_file):
    data_processing.NvidiaSmiDmonDataProcessor(dmon_data_file).get_average_data(preprocess_file=True,
                                                                                cwd="/home/ubuntu/phd_data/code/GPUETM/GPUETM")


dmon_qeaware = pd.read_csv(dmon_data_file_qeaware)
jobs_data_qeaware = pd.read_csv(jobs_data_qeaware)

dmon_max = pd.read_csv(dmon_data_file_max)
jobs_data_max = pd.read_csv(jobs_data_max)

dmon_default = pd.read_csv(dmon_data_file_default)
jobs_data_default = pd.read_csv(jobs_data_default)


def prediction_execution_time_plot():
    h = 8
    w = 6
    fig, ax = plt.subplots(figsize=(h, w))
    # pred_time = jobs_data_qeaware['']
    df = jobs_data_qeaware.drop('Unnamed: 0', 1)
    smclock = df.loc[6, :]
    # sm_clock = (df['sm_clock'].values)
    # time = (df['time'].values)
    # power = (df['pwr'].values)
    # gtemp = (df['gtemp'].values)
    # mtemp = (df['mtemp'].values)
    # smutil = (df['sm'].values)
    # memutil = (df['mem'].values)
    plt.plot(smclock, marker='*', color='blue', label="s")

    # plt.plot(sm_clock, smutil, marker='x', color='red', label='sm_util')
    # plt.plot(sm_clock, data2, marker='x', color='red', label=label2)
    # plt.plot(x, Granite_Temperature, marker='+', color='black', label='GRANITE')
    plt.legend(loc='upper right')
    plt.xlabel("Clocks (in MHz)")

    plt.ylabel("Power (in watts)")
    # plt.savefig("output/{}_{}_{}.pdf".format(application, label1, label2))

    plt.show()


def energy_bar_plot():
    N = 1

    ind = np.arange(N)  # the x locations for the groups
    width = 0.40  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 4))

    ##RMSE Old version color
    # rects1 = ax.bar(1, regression[1], width, color='darkblue', edgecolor='black', hatch='xx')
    # rects2 = ax.bar(2, regression_br[1], width, color='blueviolet', edgecolor='black', hatch='oo')
    # rects3 = ax.bar(3, regression_sgd[1], width, color='purple', edgecolor='black', hatch='++')
    # rects4 = ax.bar(4, regression_ridge[1], width, color='tomato', edgecolor='black', hatch='//')
    # rects4 = ax.bar(5, mlp[1], width, color='white', edgecolor='steelblue', hatch='--')
    # rects4 = ax.bar(6, xgb[1], width, color='white', edgecolor='magenta', hatch='..')

    # RMSE New version color

    color = 'gray'
    ##power
    y = [dmon_default.pwr.mean(), dmon_max.pwr.mean(), dmon_qeaware.pwr.mean()]
    rects1 = ax.bar(1, y[0], width, color=color, hatch='oo')
    rects2 = ax.bar(2, y[1], width, color=color, hatch='++')
    # rects3 = ax.bar(3, rf[2], width, color=color,  hatch='//')
    rects4 = ax.bar(3, y[2], width, color=color, hatch='xx')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Power (w)", fontsize=12)
    ax.set_xlabel("Scheduling policies", fontsize=12)
    # ax.set_yscale('log')

    ax.set_xticks((1, 2, 3))
    ax.set_xticklabels(('default_clock', 'max_clock', 'D-DVFS'), fontsize=10)

    # create a list to collect the plt.patches data ---------------------------------------------
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
        ax.text(i.get_x() + 0.10, i.get_height() + 0.09, str(round((i.get_height()), 2)), fontsize=10,
                color='dimgrey')

    plt.savefig("prediction_models/scheduling_power.pdf")

    plt.show()


max_exec = jobs_data_max.loc[9]
max_exec1 = max_exec[1:]
max_exec_list = pd.Series.tolist(max_exec1)

default_exec = jobs_data_default.loc[9]
default_exec1 = default_exec[1:]
default_exec_list = pd.Series.tolist(default_exec1)

dvfs_exec = dmon_qeaware.loc[9]
dvfs_exec1 = dvfs_exec[1:]
dvfs_exec_list = pd.Series.tolist(dvfs_exec1)

deadline = dmon_qeaware.loc[2]
deadline1 = deadline[1:]
deadline_list = pd.Series.tolist(deadline1)


def time_plot():
    h = 8
    w = 6
    fig, ax = plt.subplots(figsize=(h, w))

    plt.plot(max_exec_list, marker='*', color='magenta', label="max")
    plt.plot(default_exec_list, marker='*', color='red', label="default")
    plt.plot(dvfs_exec_list, marker='*', color='blue', label="d-dvfs")
    plt.plot(deadline_list, marker='*', color='black', label="deadline")
    # plt.plot(sm_clock, smutil, marker='x', color='red', label='sm_util')
    # plt.plot(sm_clock, data2, marker='x', color='red', label=label2)
    # plt.plot(x, Granite_Temperature, marker='+', color='black', label='GRANITE')
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")

    # plt.ylabel("Power (in watts)")
    # plt.savefig("output/{}_{}_{}.pdf".format(application, label1, label2))

    # plt.show()

    plt.savefig("prediction_models/exectime.pdf")

    plt.show()


prediction_execution_time_plot()

# energy_bar_plot()
# time_plot()
# dmon_data_file_qeaware = dir + "qe_aware_data_driven_dvfs_scheduler/" + "all_job_nvidiasmidmon" + "/qe_aware_data_driven_dvfs_scheduler_dmon_data"
# dmon_data_file_default = dir + "default_application_clocks_scheduler/" + "all_job_nvidiasmidmon" + "/default_application_clocks_scheduler_dmon_data"
# dmon_data_file_max
# preprocess_dmon_data(dmon_data_file_max)
