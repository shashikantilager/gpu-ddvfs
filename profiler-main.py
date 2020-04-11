import configparser
import json
import logging
import os
import time

from src import nvmlgpu
from src import workload_executor
from src.utils import executor


def collect_metric(deviceId, interval, output_folder, metric_file_name):
    """This function creates a thread that will execute the script to collect system metrics"""
    cwd = os.getcwd()
    script_path = os.path.join(cwd + "/etc/scripts/collect_gpu_metrics.sh")
    out_directory = output_folder
    cmd = list()
    cmd.append(script_path)
    cmd.append(str(deviceId))  # Device Id
    cmd.append(str(interval))  # Monitoring Interval
    cmd.append(out_directory)  # Directory for the output
    cmd.append(metric_file_name)  # File name prefix for the experiment
    print("Collecting metrics , folder " + output_folder + " file- " + metric_file_name)
    executor.create_job(cmd, background=True)
    print("Collecting metrics finished")


def kill_backgroud_process():
    """Kill the running backgound process"""
    cwd = os.getcwd()
    path = os.path.join(cwd, "etc/scripts/kill_processes.sh")
    # out_directory = output_folder + "/logs/"
    cmd = list()
    cmd.append(path)
    print("killing background processes")
    executor.create_job(cmd, background=True)
    print("killing background processes finished")


def execute_workload(application, workload_settings, run_type, folder, current_clock_set):
    workload_executor_service = workload_executor.WorkloadExecutor(logger, application, workload_settings,
                                                                   run_type, folder, current_clock_set)
    workload_executor_service.start()


### load the config files
cwd = os.getcwd()
config = configparser.ConfigParser(delimiters=("="))
path = os.path.join(cwd, "etc/configs/config.ini")
config.read(path)

###configure the logging

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger('gpuetm')
logger.setLevel(logging.INFO)
log_folder = cwd + "/output" + "/" + "logs" + "/"
## TODO - update with app and clock name
file_name = "run.log"
logfile = log_folder + file_name

if not os.path.exists(logfile):
    f = open(logfile, "w+")
    f.close()
fh = logging.FileHandler(logfile)
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

########## start application

application = config["WORKLOADS"]["application"]
workload_settings = json.load(open(cwd + "/etc/configs/" + "workload_settings.json"))
runfile = workload_settings[application]['runfile']

deviceId = int(config["GPU"]["deviceId"])
gpu_manager = nvmlgpu.GPUClockManagement(deviceId)
gpu_manager.print_clock_info()

clock_data = json.load(open(cwd + "/etc/configs/" + "gpu_supported_clocks.json"))
gpu_name = config["GPU"]['name']
supported_application_clocks = clock_data[gpu_name]['supported_clocks']

stat_history = dict()
# stat_history["Application"] = "{},{},{}".format("Mem_Clock","SM_Clock", "ExecutionTime")

folder = os.getcwd() + "/" + "output/metrics/" + application
# if not os.path.exists(folder):
#   os.makedirs(folder)

run_type = config["GPU"]["run_type"]
output_profile_folder = folder + "/" + run_type
if not os.path.exists(output_profile_folder):
    os.makedirs(output_profile_folder)

counter = 0

for clock_sets in supported_application_clocks:
    for graphics_clock in clock_sets['graphics']:
        counter = counter + 1
        if counter % 2 == 0 and graphics_clock < 890:
            # if 1 ==1: ## temporary
            print("counter value: {}, Executing the workload for  the clock set {} : {}".format(counter,
                                                                                                clock_sets['memory'],
                                                                                                graphics_clock))
            gpu_manager.set_application_clocks(clock_sets['memory'], graphics_clock, deviceId)

            ### start collecting metrics
            interval = int(config['GPU']['dmon_interval'])

            current_clock_set = str(clock_sets['memory']) + "_" + str(graphics_clock)

            metric_file_name = application + "_" + current_clock_set
            is_collect_metric = config["INPUT_OUTPUT"]["collect_metric"]

            if is_collect_metric == "True":
                dmon_folder = output_profile_folder + "/" + "nvidiasmidmon"
                if not os.path.exists(dmon_folder):
                    os.makedirs(dmon_folder)
                collect_metric(deviceId, interval, dmon_folder, metric_file_name)
            # This sleep function is to allow dmon thread to start and reach steady state before starting the application
            time.sleep(2)
            ### Workload Executor
            start_time = time.time()

            print("execute workload start")
            run_folder = output_profile_folder + "/" + "run_time_data"
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)
            execute_workload(application, workload_settings, run_type, run_folder, current_clock_set)

            end_time = time.time()

            total_time = end_time - start_time

            stat_history[metric_file_name] = "{},{},{}".format(clock_sets['memory'], graphics_clock, total_time)
            print("execute workload done")
            ## sleep for 2 secs to allow dmon thread to reach steady state in its data collection . Also lowest record interval
            # is 1 sec, which requires some time to log the the data after workload execution
            time.sleep(2)

            # Kill metric collector thread

            print("Starting application kill")
            kill_backgroud_process()
            ##make main thread sleep for 2 seconds before starting new execution
        # time.sleep(2)

        else:
            print("counter value: {}, skipping the clock set {} : {}".format(counter, clock_sets['memory'],
                                                                             graphics_clock))
#############

# print time  to file

file_name_stat = os.path.join(output_profile_folder + "/" + "stat_histroy" + ".csv")

with open(file_name_stat, 'w') as f:
    for key, value in stat_history.items():
        f.write("%s,%s\n" % (key, value))

##TODO kill metric collector, check its necessity
print("Starting application kill")
kill_backgroud_process()
print("Finished application kill")

################## Joining All the Bits of Data into One file
## TODO, read directly from dictionary
# df = pd.read_csv(file_name_stat, squeeze=True)

# ## For each clock set we have executed for a  application,  combine the data from dmon, nsight and our stat history
# for index, row in df.iterrows():
#     # print("index: {} , data: {} ".format(index, row[0]))
#     clock_set = "{}_{}".format(row[1], row[2])
#     timetaken = np.array([row[3]])
#     time = pd.Series(timetaken, index=['time'])
#     print(clock_set)
#     metricfolder = os.getcwd() + "/" + "output/metrics/" +  application
#     dmondatafolder = metricfolder + "/" + "nvidiasmidmon/"
#     nsightdata = metricfolder + "/" + "nsightprofile/"
#     combined_data = metricfolder + "combined_average_data/"
#
#     dmondatafile = dmondatafolder + application + "_" + clock_set
#     nsightdatabase = nsightdata + application + "_nsys_" + clock_set + ".sqlite"
#
#     # If the output file required to converto to standard csv, to load into dataframe, set preprocess_file to True
#     dmon_data_processor = data_processing.NvidiaSmiDmonDataProcessor(dmondatafile).get_average_data(preprocess_file=True)
#
#     nsight_data_processor = data_processing.NSightProfileDataProcessor(nsightdatabase).get_averaged_data_from_tables()
#
#     # combine all three Series data into one
#     final_data = pd.concat([nsight_data_processor, dmon_data_processor, time])
#
#     output_dir = metricfolder + "combined_average_data/"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     output_file = output_dir + application + "_" + clock_set
#     final_data.to_csv(output_file)
#
