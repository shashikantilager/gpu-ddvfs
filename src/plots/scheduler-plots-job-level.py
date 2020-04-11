import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data_processing

# dir =  "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/schedule-data/"
job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/joblevel_w_50_kmeans/"

# job_dir = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/newscheduledata/scheduler/alphadata/syslevel_w_50_alpha_2/"

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


def preprocess_dmon_data(dmon_data_folder):
    for app in application_que:
        print (app)
        dmondata = dmon_data_folder + app
        data_processing.NvidiaSmiDmonDataProcessor(dmondata).get_average_data(preprocess_file=True,
                                                                              cwd="/home/ubuntu/phd_data/code/GPUETM/GPUETM")


# def combinedmondata(dmon_data_folder):
def print_dmon_data():
    ##get this que from execution sequence
    application_que = ['CORR', 'particlefilter_naive', 'COVAR', 'GEMM', 'ATAX', 'myocyte', 'SYR2K',
                       'particlefilter_float',
                       'SYRK', '2MM', 'backprop', 'lavaMD']
    # preprocess_dmon_data(dmon_data_file_max)
    # preprocess_dmon_data(dmon_data_file_qeaware)
    # preprocess_dmon_data(dmon_data_file_default)
    print("{},{},{},{}".format("App", "Max", "Default", "QEAware"))
    dmon_data_sum = {}
    dmon_data_avg = {}

    for app in application_que:
        # print (app)

        df_max = pd.read_csv(dmon_data_file_max + app)
        df_default = pd.read_csv(dmon_data_file_default + app)
        df_qeawre = pd.read_csv(dmon_data_file_qeaware + app)
        print ("{},{},{},{}".format(app, df_max.pwr.sum(), df_default.pwr.sum(), df_qeawre.pwr.sum()))
        # print ("{},{},{},{}".format(app, df_max.pwr.mean(),df_default.pwr.mean(),df_qeawre.pwr.mean()))


# print_dmon_data()
jobs_data_qeaware = pd.read_csv(jobs_data_qeaware)
#
# dmon_max =  pd.read_csv(dmon_data_file_max)
jobs_data_max = pd.read_csv(jobs_data_max)
#
#
# dmon_default =  pd.read_csv(dmon_data_file_default)
jobs_data_default = pd.read_csv(jobs_data_default)


# --------------------------Print Dmon
# print_dmon_data()
#
# # ------------------------------- plot job level energy
# df = pd.read_csv("/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data/job_level_dmon_sum_50.csv")
# for index, row in df.iterrows():
#     # App, Max, Default, QEAware
#     fig, ax = plt.subplots()
#
#     print(row['App'], row['Max'], row['Default'], row['QEAware'])# dmon_qeaware =  pd.read_csv(dmon_data_file_qeaware)
#     objects = ('Max Clock', 'Default Clock', 'D-DVFS', )
#     y_pos = np.arange(len(objects))
#     values = [row['Max'], row['Default'], row['QEAware']]
#
#     plt.bar(y_pos, values, align='center', width= 0.2)
#     plt.xticks(y_pos, objects)
#     plt.ylabel('Power (Watts)')
#
#     # plt.title('')
#     plt.show()
#     filename = "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data-schedule-energy/" + row['App'] + "_50.pdf"
#     plt.savefig(filename, bbox_inches='tight' )


# -------------Grouped bar plot------------------------
def gropuedbarplot():
    df = pd.read_csv("/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data/job_level_dmon_avg_kmeans_w_50_RD.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    barWidth = 0.2

    # set height of bar
    bars1 = df['Max']
    bars2 = df['Default']
    bars3 = df['QEAware']
    ## new for normalisation TODO
    # np.array(df['to''tal_bedrooms'])
    # bars1 = np.array( df['Max'])
    # bars2 =np.array( df['Default'])
    # bars3 =  np.array(df['QEAware'])
    # bars1 = preprocessing.normalize(bars1)
    # bars2 = preprocessing.normalize(bars2)
    # bars3 = preprocessing.normalize(bars3)

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    color = "white"
    # Make the plot
    plt.bar(r1, bars1, color='lightcoral', width=barWidth, edgecolor='black', label='MC', hatch="//")
    plt.bar(r2, bars2, color='blue', width=barWidth, edgecolor='black', label='DC', hatch="--")
    plt.bar(r3, bars3, color='c', width=barWidth, edgecolor='black', label='D-DVFS', hatch="++")

    # Add xticks on the middle of the group bars
    plt.xlabel('Applications', fontsize=14)
    plt.ylabel('Average Power (Watts)', fontsize=14)
    plt.xticks([r + barWidth for r in range(len(bars1))], df['App'], rotation=45, ha='right', fontsize=10)
    # ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)

    # for i in ax.patches:
    #     # get_x pulls left or right; get_height pushes up or down
    #     # use + or - to adjust the postion as shown below
    #     # ax.text(i.get_x() + 0.10, i.get_height() + 0.04, str(round((i.get_height()), 2)), fontsize=10,
    #     ax.text(i.get_x() + 0.04, i.get_height(), str(round((i.get_height()), 2)), fontsize=8,
    #             color='black')
    # Create legend & Show graphic
    plt.legend()
    filename = "/home/ubuntu/phd_data/code/GPUETM/GPUETM/src/plots/data-schedule-energy/" + "energgygroupedbarplot_kmeans" + "w_50_avg_RD.pdf"
    plt.savefig(filename, bbox_inches='tight')
    #
    plt.show()


gropuedbarplot()
