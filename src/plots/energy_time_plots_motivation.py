import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns


# file_name_stat = "/home/shashi/gpu/g5kcopydata/streamcluster/stat_histroy.csv"
# file_name_stat = "/home/shashi/gpu/g5kcopydata/particlefilter/stat_histroy.csv"
# /home/shashi/gpu/g5kcopydata/particlefilter


# application = "streamcluster"
# application = "particlefilter"
# metricfolder = "/home/shashi/gpu/g5kcopydata" + "/" +  application


#
# ## Move this to data processing along with combining the data part
# def get_total_dataframe():
#
#     total_df = pd.DataFrame()
#     for index, row in stat_df.iterrows():
#         # print("index: {} , data: {} ".format(index, row[0]))
#         mem_clock = row[1]
#         sm_clock = row[2]
#
#         clock_set = "{}_{}".format(mem_clock, sm_clock)
#         timetaken = np.array([row[3]])
#     #     time = pd.Series(timetaken, index=['time'])
#         print(clock_set )
#         combined_data = metricfolder + "/" + "combined_average_data/"
#         combined_data_file = combined_data + application + "_" + clock_set
#         combined_data_df = pd.read_csv(combined_data_file, header= None)
#
#          # transposing and converting to dataframe(First column now becomes row, i,e,
#         # header for the datraframe with other row as a its correpsonding values for each columns) .
#         cdt = combined_data_df.T
#         # make first row as header
#         new_header = cdt.iloc[0]
#         ## take rest of data, here next row
#         cdt = cdt[1:]
#         # Add the header back
#         cdt.columns = new_header
#         ##  Insert a Clock Info for the reference
#         cdt.insert(0, 'mem_clock', mem_clock)
#         cdt.insert(1, 'sm_clock', sm_clock)
#
#         total_df = total_df.append(cdt, ignore_index=True)
#
#     filename = metricfolder  + "/" + "total_dataframe.csv"
#     total_df.to_csv(filename, index= False)

def plot_corr(data, application, size=18):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        data: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    # Compute the correlation matrix for the received dataframe
    co_relation = data.corr()
    # for one var
    # cputemp = data[data.columns[0:]].corr()['CPU1_Temperature'][:-1]

    fig = plt.figure(figsize=(20, 20))
    h = 20
    w = 20
    fig, ax = plt.subplots(figsize=(h, w))

    # fig = plt.figure(figsize= (8,6))
    # ax = fig.add_subplot(111)
    cax = ax.matshow(co_relation, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.rcParams.update({'font.size': 12})
    plt.xticks(fontsize=12, rotation=90)
    ax.set_yticks(ticks)
    plt.yticks(fontsize=12)
    ax.set_xticklabels(data.columns)

    ax.set_yticklabels(data.columns)

    plt.savefig(os.path.join("output" + "/correlationTeslaP100" + application + ".pdf"), bbox_inches='tight', ax=ax)
    plt.show()


# Two pass clustering
# 1-We cluster the corr matrix
#   We sort the survey data according to this clustering
# 2-For cluster bigger than a threshold we cluster those sub-clusters
#   We sort the survey data according to these clustering


def two_pass_clustering_and_visualize(df):
    cluster_th = 4

    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5 * d.max(), 'distance')

    columns = [df.columns.tolist()[i] for i in list(np.argsort(ind))]
    df = df.reindex_axis(columns, axis=1)

    unique, counts = np.unique(ind, return_counts=True)
    counts = dict(zip(unique, counts))

    i = 0
    j = 0
    columns = []
    for cluster_l1 in set(sorted(ind)):
        j += counts[cluster_l1]
        sub = df[df.columns.values[i:j]]
        if counts[cluster_l1] > cluster_th:
            X = sub.corr().values
            d = sch.distance.pdist(X)
            L = sch.linkage(d, method='complete')
            ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
            col = [sub.columns.tolist()[i] for i in list((np.argsort(ind)))]
            sub = sub.reindex_axis(col, axis=1)
        cols = sub.columns.tolist()
        columns.extend(cols)
        i = j
    df = df.reindex_axis(columns, axis=1)

    plot_corr(df, 18)


def create_histogram(df):
    # plt.hist(x, weights=np.ones_like(x)/1000, color='cadetblue', edgecolor='black', bins=int(180 / 5),  alpha =0.3)

    #
    sm_clock = (df['sm_clock'].values)
    time = (df['time'].values)
    power = (df['pwr'].values)
    gtemp = (df['gtemp'].values)
    mtemp = (df['mtemp'].values)
    smutil = (df['sm'].values)
    memutil = (df['mem'].values)
    x = gtemp
    plt.figure(figsize=(8, 6))

    # h = 8
    # w = 6
    # fig, ax = plt.subplots(figsize=(h, w))

    plt.rcParams.update({'font.size': 18})  # ALL FONTS INCLUDING TICKS LABLELS WILL BE SETWITH THIS
    sns.distplot(x, color="black", bins=int(180 / 5))  #
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)

    plt.xlabel('CPU Temperature ($^\circ$C)')
    # plt.ylabel('Number of Data Points (X1000)')
    plt.ylabel('Density')
    # plt.savefig('../Plots/Motivation-plots/CPUtemperatureHistogramDensityrevised2.pdf', bbox_inches='tight')
    # plt.savefig('../Plots/Motivation-plots/CPUtemperatureHistogramDensityrevised4.pdf', bbox_inches='tight')

    plt.show()


def generic_plots(df):
    h = 8
    w = 6
    fig, ax = plt.subplots(figsize=(h, w))
    sm_clock = (df['sm_clock'].values)
    time = (df['time'].values)
    power = (df['pwr'].values)
    gtemp = (df['gtemp'].values)
    mtemp = (df['mtemp'].values)
    smutil = (df['sm'].values)
    memutil = (df['mem'].values)
    data1 = time
    data2 = power
    label1 = "Time"
    label2 = "Power"
    plt.plot(sm_clock, data1, marker='*', color='blue', label=label1)
    # plt.plot(sm_clock, smutil, marker='x', color='red', label='sm_util')
    # plt.plot(sm_clock, data2, marker='x', color='red', label=label2)
    # plt.plot(x, Granite_Temperature, marker='+', color='black', label='GRANITE')
    plt.legend(loc='upper right')
    plt.xlabel("Clocks (in MHz)")

    plt.ylabel("Power (in watts)")
    plt.savefig("output/{}_{}_{}.pdf".format(application, label1, label2))

    plt.show()


def double_Yaxes_plot(df, data1="pwr", data2="time", label1="power", label2="time", savefig=False):
    sm_clock = (df['sm_clock'].values)
    # time = (df['time'].values)
    # power = (df['pwr'].values)
    # gtemp = (df['gtemp'].values)
    # # mtemp = (df['mtemp'].values)
    # smutil = (df['sm'].values)
    # memutil = (df['mem'].values)

    fig, ax1 = plt.subplots()

    x = sm_clock
    data1 = (df[data1].values)

    data2 = (df[data2].values)

    color = 'tab:red'
    ax1.set_xlabel("SM Clock (s)")
    ax1.set_ylabel(label1, color=color)
    ax1.plot(x, data1, marker='*', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label2, color=color)  # we already handled the x-label with ax1
    ax2.plot(x, data2, marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if savefig:
        output_dir = os.getcwd() + "/" + application + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_name = output_dir + "{}_{}_{}.pdf".format(application, label1, label2)
        plt.savefig(output_file_name)
        plt.show()


def applications_plot():
    directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/directrunmetrics/"
    apps = "2MM  ATAX  CORR  COVAR  GEMM  myocyte  particlefilter_float  particlefilter_naive  srad_v1  streamcluster  SYR2K  SYRK"
    application_list = list(apps.split("  "))

    for application in application_list:
        metricfolder = directory + application
        # file_name_stat = metricfolder + "/" + "stat_histroy.csv"
        # # stat_df = pd.read_csv(file_name_stat, squeeze=True, header= None)
        # # stat_df.columns = ['Application', 'MemClock' , 'SMClock', 'Time']
        # # stat_df = stat_df.sort_values(by=['SMClock'])
        #
        #

        # total_df = pd.read_csv(os.path.join(metricfolder + "/total_dataframe.csv" ))
        # ## TODO avoid the insertion
        total_df = pd.read_csv(os.path.join(metricfolder + "/total_dmon_dataframe.csv"))

        # total_df = total_df.drop(total_df.columns[0], axis= 1)

        # two_pass_clustering_and_visualize(total_df)
        # plot_corr(total_df, application)
        # generic_plots(total_df)

        # double_Yaxes_plot(df, data1= "pwr", data2 ="time", label1= "power", label2 ="time"):
        # time = (df['time'].values)
        # power = (df['pwr'].values)
        # gtemp = (df['gtemp'].values)
        # # mtemp = (df['mtemp'].values)
        # smutil = (df['sm'].values)
        # memutil = (df['mem'].values)

        # double_Yaxes_plot(total_df)
        double_Yaxes_plot(total_df, data2="mem", label2="Mem_Util", savefig=True)
        # create_histogram(total_df)


directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"
file = directory + "final_data.csv"
total_df = pd.read_csv(file)
# apps = "2MM  ATAX  CORR  COVAR  GEMM  myocyte  particlefilter_float  particlefilter_naive  srad_v1  streamcluster  SYR2K  SYRK"
apps = "2MM,ATAX,CORR,COVAR,GEMM,myocyte,particlefilter_float,particlefilter_naive,srad_v1,streamcluster,SYR2,SYR,backprop,lavaMD"
apps = apps.split(sep=",")
#
# # -------------------------Power Time-----------power clock- at kernel level---------------------------------------------------
# apps =["CORR","COVAR"]
p = []
c = []
for app in apps:

    data = total_df[(total_df["Application_name"] == app)]
    # data = data[data['Kernel_Name'] == 'mm2_kernel2(float* float* float*)'] # 2MM
    kernel_names = data.Kernel_Name.unique()

    for kernel in kernel_names:
        print(kernel)
        # get kernel name excluding parameters
        kernelname = kernel.split(sep='(')[0]
        kerneldata = data[data['Kernel_Name'] == kernel]

        pwr = kerneldata.pwr
        time = kerneldata.time
        pwr = np.array(pwr)
        time = np.array(time)

        clock = kerneldata.sm_clock
        clock = np.array(clock)
        # clock = sorted(time)

        # clock_smooth = np.linspace(clock.min(), clock.max(),  clock.__len__())
        #
        # spl = make_interp_spline(time, pwr, k=3)  # type: BSpline

        label = app + "_" + kernelname

        if label == "CORR_reduce_kernel":
            p = pwr
            c = clock

            # plt.plot(clock, pwr, marker='*', color='blue', label="pwr")
            # plt.plot(clock, pwr, color='red', label="Power")
            # plt.plot(clock, time, color='blue', label="Power")
            # plt.plot(time, pwr, color='black', label="Power")

            # plt.plot(time, pwr, marker='*', label=label) ### old one
            # ax1.scatter(x, pwr, label = kernelname)
            # plt.legend(loc='upper right')

            plt.xlabel("Time (seconds)")
            plt.ylabel("Power (Watts)")

            # plt.savefig("power_time_output/" +   label + ".pdf")

            # plt.savefig("power_clock/" +   label + ".pdf")
            # plt.savefig("power_time_output/" +   label + ".pdf")

            plt.show()

# ---------------------------------double y axes kernel level---------------------------
# apps = ["2MM", "ATAX"]

#
# for app in apps:
#
#     data = total_df[(total_df["Application_name"] == app)]
#     # data = data[data['Kernel_Name'] == 'mm2_kernel2(float* float* float*)'] # 2MM
#     kernel_names = data.Kernel_Name.unique()
#
#     for kernel in kernel_names:
#         print(kernel)
#         # get kernel name excluding parameters
#         kernelname = kernel.split(sep='(')[0]
#         kerneldata = data[data['Kernel_Name'] == kernel]
#         pwr = pd.Series.tolist(kerneldata.pwr)
#         time = pd.Series.tolist(kerneldata.sm_clock)
#         # plt.plot(time, pwr, marker='*', color='blue', label="pwr")
#         label = app + "_" + kernelname
#         fig, ax1 = plt.subplots()
#
#         x = (kerneldata['sm_clock'].values)
#         data1 = (kerneldata['pwr'].values)
#
#         data2 = (kerneldata['time'].values)
#
#
#         color = 'red'
#         ax1.set_xlabel("SM Clock (s)")
#         ax1.set_ylabel("Power (watts)", color=color)
#         # ax1.plot(x, data1, marker='*',color=color)
#
#         ##amooth curve part
#         # length epresents number of points to make between T.min and T.max
#         # x = np.linspace(x.min(), x.max(), x.__len__())
#         # spl = make_interp_spline(x, data1, k=3)  # type: BSpline
#         # data1 = spl(x)
#
#
#         ax1.plot(x, data1,color=color,  label = "Power")
#         ax1.tick_params(axis='y', labelcolor=color)
#         # plt.legend(loc='upper  right')
#
#         ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#         color = 'blue'
#         ax2.set_ylabel("Time (seconds)", color=color)  # we already handled the x-label with ax1
#         # ax2.plot(x, data2, marker='x',color=color)
#
#         ##smooth curve part
#         # spl = make_interp_spline(x, data2, k=3)  # type: BSpline
#         # data2 = spl(x)
#
#
#
#         ax2.plot(x, data2, color=color, label = "Time")
#         ax2.tick_params(axis='y', labelcolor=color)
#         # plt.legend(loc='upper left')
#         # added these three lines
#         # ax1.legend(loc=0)
#         ###Adding legends manually based on required positions by specifying x and y
#         ax1.legend(loc=(.3, .93), frameon=False)
#         ax2.legend(loc=(.5, .93), frameon=False)
#         fig.tight_layout()  # otherwise the right y-label is slightly clipped
#
#         plt.savefig("power_clock_time/" +   label + ".pdf")
#         plt.show()
