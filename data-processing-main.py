import configparser
import fnmatch
import json
import os
import random

import numpy as np
import pandas as pd

from src import data_processing

### load the config files
cwd = os.getcwd()
config = configparser.ConfigParser(delimiters=("="))
path = os.path.join(cwd, "etc/configs/config.ini")
config.read(path)

## This script requires all the data to be moved from profile run and direct run outputs(output/metrics/*) to the directory specified in the config file
data_directory = config["INPUT_OUTPUT"]["data_directory"]


def process_dmon_data():
    directory = data_directory + "directrunmetrics/"
    # apps = "2MM  ATAX  CORR  COVAR  GEMM  myocyte  particlefilter_float  particlefilter_naive  srad_v1  streamcluster  SYR2K  SYRK"
    apps = "myocyte"
    # for p100 test data
    # directory = data_directory +    "nvprofmetrics/metrics/"
    # apps = "2MM  ATAX  CORR  COVAR  GEMM  particlefilter_float  particlefilter_naive  backprop lavaMD SYR2K  SYRK"

    application_list = list(apps.split("  "))

    for application in application_list:

        application_dir = directory + application

        file_name_stat = application_dir + "/" + "stat_histroy.csv"
        #
        # ######### Joining All the Bits of Data into One file
        # # TODO, read directly from dictionary
        df = pd.read_csv(file_name_stat, squeeze=True, header=None)
        #
        total_df = pd.DataFrame()
        # ## For each clock set we have executed for a  application,  combine the data from dmon, nsight and our stat history
        for index, row in df.iterrows():
            # print("index: {} , data: {} ".format(index, row[0]))

            clock_set = "{}_{}".format(row[1], row[2])

            timetaken = np.array([row[3]])

            time = pd.Series(timetaken, index=['time'])

            clocks = pd.Series([row[1], row[2]], index=['mem_clock', 'sm_clock'])
            print(clock_set)

            dmon_data_folder = application_dir + "/" + "nvidiasmidmon/"

            dmon_data_file = dmon_data_folder + application + "_" + clock_set

            avg_dmon_data = data_processing.NvidiaSmiDmonDataProcessor(dmon_data_file).get_average_data(
                preprocess_file=True)

            ##Insert time data and clock data from stat_history to the dmon data for the reference
            dmon_df = pd.concat([avg_dmon_data, clocks, time])

            output_dir = application_dir + "/" + "dmon_average_data/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = output_dir + application + "_" + clock_set
            ## As its a average data, this dataframe has one row, i.e, stored as Series
            dmon_df.to_csv(output_file)

            # transposing and converting Series data to dataframe(First index column now becomes header row, i,e,
            # The remaining one column is transformed as  second row, i.e, corresponding values for each columns) .

            dmon_df = dmon_df.T
            #
            # make first row as header
            new_header = dmon_df.iloc[0]
            ## take rest of data, here next row
            dmon_df = dmon_df[1:]
            # Add the header back
            dmon_df.columns = new_header

            total_df = total_df.append(dmon_df, ignore_index=True)

        filename = application_dir + "/" + "total_dmon_dataframe.csv"
        total_df.to_csv(filename, index=False)


def process_nvprof_data():
    directory = data_directory + "nvprofmetrics/metrics/"
    # apps = "2MM  ATAX  CORR  COVAR  GEMM  myocyte  particlefilter_float  particlefilter_naive  SYR2K  SYRK"
    # apps  = "2MM"
    # apps = "2MM  ATAX  CORR  COVAR  GEMM  particlefilter_float  particlefilter_naive  SYR2K  SYRK"
    apps = "backprop"
    # apps = "2MM  ATAX  CORR  COVAR  GEMM  particlefilter_float  particlefilter_naive  myocyte backprop lavaMD SYR2K  SYRK"
    # application_list= list(apps.split("  "))
    application_list = apps.replace(' ', '').split(",")
    procesed_data = pd.DataFrame()

    for application in application_list:

        ## For old folder structure
        # application_dir = directory + application
        ## for new folder structure
        application_dir = directory + application + "/nvprof_profiler"

        file_name_stat = application_dir + "/" + "stat_histroy.csv"
        # # TODO, read directly from dictionary
        df = pd.read_csv(file_name_stat, squeeze=True, header=None)

        for index, row in df.iterrows():
            print("index: {} , data: {} ".format(index, row[0]))

            clock_set = "{}_{}".format(row[1], row[2])  # row[1] is mem clock and row [2] is sm_clock

            print("Current Clock Set: {}".format(clock_set))

            ## For old folder structure
            # nvprof_data_folder = application_dir + "/" + "nvprof/"
            ## for new folder structure
            nvprof_data_folder = application_dir + "/" + "run_time_data/"

            nvprof_data_file = None
            ##Get the file name of particular clock set (as file name as different process IDs, we select based on regular expression matching clock set)
            for file in os.listdir(nvprof_data_folder):
                print("file:--- {}".format(file))
                if fnmatch.fnmatch(file, '*_{}'.format(clock_set)):
                    nvprof_data_file = nvprof_data_folder + file

            if nvprof_data_file is not None:
                print("Processing NVProf File:{}".format(nvprof_data_file))
                nvprof_data = data_processing.NVProfProfilerDataProcessor(nvprof_data_file).get_data()

                kernels = nvprof_data.Kernel.unique()
                nvprof_data = nvprof_data.drop(["Metric Description", "Min", "Max"], axis=1)

                for i in range(0, kernels.size):
                    print("Kernels: {}".format(kernels[i]))
                    kernel_data = nvprof_data[nvprof_data.Kernel == kernels[i]]
                    ## Take the header, feature information
                    header_values = kernel_data['Metric Name']

                    kernel_data = kernel_data.drop(["Metric Name"], axis=1)
                    kernel_data = kernel_data.transpose()

                    kernel_data.columns = header_values

                    kernel_data.insert(loc=0, column='Kernel_Name', value=kernels[i])
                    kernel_data.insert(loc=1, column='mem_clock', value=row[1])
                    kernel_data.insert(loc=2, column='sm_clock', value=row[2])

                    kernel_data = pd.DataFrame(kernel_data, index=['Avg'])

                    procesed_data = procesed_data.append(kernel_data)



            else:
                print("NVProf file not found for clock set:{}".format(clock_set))

        ##Write all the clock set files processed data to single file
        filename = application_dir + "/" + "total_nvprof_dataframe.csv"
        procesed_data.to_csv(filename, index=False)


def combine_dmon_and_nvprof_data():
    nvprof_directory = data_directory + "nvprofmetrics/metrics/"

    dmon_directory = data_directory + "directrunmetrics/"

    # for p100 test datta
    # dmon_directory = data_directory + "nvprofmetrics/metrics/"

    apps = "2MM,ATAX,CORR,COVAR,GEMM,particlefilter_float,particlefilter_naive,lavaMD,backprop,myocyte,SYR2K,SYRK"
    # apps  = "lavaMD,backprop,myocyte"
    application_list = list(apps.split(","))
    print(application_list.__len__())
    final_data = pd.DataFrame()
    for application in application_list:
        print("Processing Application:{}".format(application))
        #
        # file_name_stat = dmon_directory + "/" + "stat_histroy.csv"
        # # # TODO, read directly from dictionary
        # df = pd.read_csv(file_name_stat, squeeze=True, header=None)

        ##old file structure
        total_nvprof_data_file = nvprof_directory + application + "/" + "total_nvprof_dataframe.csv"
        # new folder structure
        # total_nvprof_data_file = nvprof_directory + application + "/" + "nvprof_profiler/" +"total_nvprof_dataframe.csv"

        total_dmon_data_file = dmon_directory + application + "/" + "total_dmon_dataframe.csv"
        nvprof_df = pd.read_csv(total_nvprof_data_file)
        dmon_df = pd.read_csv(total_dmon_data_file)

        combined_df = pd.merge(nvprof_df, dmon_df, on='sm_clock', how='left')
        ##remove redudant column inserted from dmon frame
        # combined_df = combined_df.drop(columns = ['mem_clock_y'], axis =1)
        combined_df.insert(loc=0, column='Application_name', value=application)
        ##Write all the clock set files processed data to single file
        ## TODO - Error no file found
        filename = data_directory + "combined_metrics" + "/" + application + "_dmon_nvprof_combined.csv"
        combined_df.to_csv(filename, index=False)

        final_data = final_data.append(combined_df)

        # Remove Units exist in column names
        final_data[final_data.columns] = final_data[final_data.columns].replace(
            {'%': '', ',': '', '/s': '', 'GB': '', 'MB': '', 'B': '', 'K': ''}, regex=True)

    final_data_file = data_directory + "final_data.csv"
    final_data.to_csv(final_data_file)
    print("All applications combined data is written to file : {}".format(final_data_file))


# Fiilters deafult application clock data into seperate file. Which is useful while scheduling to get the feature list
def parse_deafault_application_clock_data():
    clock_data = json.load(open(os.getcwd() + "/etc/configs/" + "gpu_supported_clocks.json"))
    gpu_name = config["GPU"]['name']
    print(gpu_name)
    print(clock_data[gpu_name])
    mem_lock, sm_clock = clock_data[gpu_name]['default_application_clock']
    print("Default Clock for GPU: {} are Mem: {} SM: {}".format(gpu_name, mem_lock, sm_clock))
    total_data_file = data_directory + "final_data.csv"
    total_df = pd.read_csv(total_data_file)

    default_clock_data = total_df[(total_df["mem_clock_x"] == mem_lock) & (total_df["sm_clock"] == sm_clock)]
    default_clock_data_file = data_directory + "default_clock_data.csv"
    default_clock_data.to_csv(default_clock_data_file)
    return default_clock_data


# This will get the deadline for each application based on uniform distribution
def get_deadline(start, end):
    return random.uniform(start, end)


## arrival time of applications
def get_arrival_time():
    return random.uniform(1, 50)


def get_exec_deadline_arrival_time():
    application_que = "2MM,ATAX,CORR,COVAR,GEMM,particlefilter_float,particlefilter_naive,lavaMD,backprop,myocyte,SYR2K,SYRK"
    application_list = application_que.replace(' ', '').split(",")
    app_time_deadline_info = dict()

    clock_data = json.load(open(os.getcwd() + "/etc/configs/" + "gpu_supported_clocks.json"))
    # workload_settings = json.load(open(os.getcwd() + "/etc/configs/" + "workload_settings.json"))
    workload_settings = json.load(open(os.getcwd() + "/etc/configs/" + "workload_settings_a_50_alpha_1.json"))
    gpu_name = config["GPU"]['name']
    mem_lock, sm_clock = clock_data[gpu_name]['default_application_clock']

    dir = data_directory + "directrunmetrics/"
    totaleaxec = 0
    for application in application_list:

        # application =  "2MM"

        application_dir = dir + application

        file_name_stat = application_dir + "/" + "stat_histroy.csv"

        df = pd.read_csv(file_name_stat, squeeze=True, header=None)

        for index, row in df.iterrows():
            # print("index: {} , data: {} ".format(index, row[0]))
            if row[1] == mem_lock and row[2] == sm_clock:
                clock_set = "{}_{}".format(row[1], row[2])
                default_clock_exec_time = row[3]
                totaleaxec += default_clock_exec_time
                # arrival = get_arrival_time()
                ## The deadline will be normal distribution having between startime and  tow times of of its default execution time

                # deadline =  get_deadline(2*default_clock_exec_time, 2.5*default_clock_exec_time)
                deadline = 3 * default_clock_exec_time
                workload_settings[application]["execution_time"] = default_clock_exec_time
                # workload_settings[application]["arrival_time"] = arrival
                arrival = workload_settings[application]["arrival_time"]

                workload_settings[application]["sys_time_deadline"] = arrival + deadline
                workload_settings[application]["exec_time_deadline"] = deadline
                # del  workload_settings[application]["deadline"]

                # app_time_deadline_info[application] = np.array(default_clock_exec_time, deadline)
                # print("App:{}, Exec: {}, Arrival: {} Deadline: {} Execdeadline {}".format(application, default_clock_exec_time, arrival, deadline,
                #                                                                          deadline -arrival)) # base profile time taken

        with open('/home/ubuntu/phd_data/code/GPUETM/GPUETM/etc/configs/workload_settings_a_50_alpha_3.json',
                  'w') as outfile:
            json.dump(workload_settings, outfile, indent=4)

    # df = pd.DataFrame.from_dict(app_time_deadline_info, orient='index')
    # # Convert index to first column
    # df.reset_index(level=0, inplace=True)
    # df.columns = ['Application', 'Exec_Time', 'Deadline']
    # df.to_csv(os.path.join(data_directory + "applicaiton_exec_deadline_info.csv"))
    # return   df
    print("TotalExec: {}".format(totaleaxec))


# process_nvprof_data()
# process_dmon_data()
# combine_dmon_and_nvprof_data()
# df = get_exec_time()

# df = parse_deafault_application_clock_data()

# deafult_data = pd.read_csv(os.path.join(data_directory + "default_clock_data.csv"))

# TODO -- insert workload setting data automaticaly- build scheduler code
# directory = "/home/ubuntu/phd_data/data/GPUETM/TeslaP100data/"
# file = directory + "final_data.csv"
# total_df = pd.read_csv(file)

##TODO - work with deafult calls and provide class methods
# parse_deafault_application_clock_data()
# default_clock_data = pd.read_csv(data_directory +"default_clock_data.csv")
get_exec_deadline_arrival_time()
