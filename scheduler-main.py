import configparser
import json
import logging
import os

import pandas as pd

from src.scheduler import default_application_clocks_scheduler
from src.scheduler import max_application_clock_scheduler
from src.scheduler import qe_aware_data_driven_dvfs_scheduler
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
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
## TODO - update woith app and clock name
file_name = "scheduler.log"
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

# application_que = ["2MM", "ATAX"]
application_que = config['SCHEDULER']['applications_que'].replace(' ', '').split(",")

print("-----{}".format(application_que))
workload_settings = json.load(open(cwd + "/etc/configs/" + "workload_settings.json"))

clock_data = json.load(open(cwd + "/etc/configs/" + "gpu_supported_clocks.json"))
gpu_name = config["GPU"]['name']

# TODO different scheduler - run each one - refer topo-aware


scheduler_name = config["SCHEDULER"]["scheduler_name"]
scheduler = None
monitor_job_level_data = True

deviceId = int(config["GPU"]["deviceId"])
data_directory = config["INPUT_OUTPUT"]["data_directory"]
folder = os.getcwd() + "/" + "output/scheduler/joblevel_w_50_kmeans/" + scheduler_name  # joblevel_w_50  syslevel_w_50
if not os.path.exists(folder):
    os.makedirs(folder)

if scheduler_name == "qe_aware_data_driven_dvfs_scheduler":
    scheduler = qe_aware_data_driven_dvfs_scheduler.QEDVFSScheduler(logger, application_que,
                                                                    workload_settings, deviceId,
                                                                    data_directory, folder, monitor_job_level_data)

elif scheduler_name == "default_application_clocks_scheduler":
    default_application_clocks = clock_data[gpu_name]['default_application_clock']
    scheduler = default_application_clocks_scheduler.DefaultApplicationClockScheduler(logger, application_que,
                                                                                      workload_settings, deviceId,
                                                                                      data_directory, folder,
                                                                                      default_application_clocks,
                                                                                      monitor_job_level_data)
elif scheduler_name == "max_application_clock_scheduler":
    max_application_clocks = clock_data[gpu_name]['max_application_clocks']
    print(max_application_clocks)
    scheduler = max_application_clock_scheduler.MaxApplicationClockScheduler(logger, application_que,
                                                                             workload_settings, deviceId,
                                                                             data_directory, folder,
                                                                             max_application_clocks,
                                                                             monitor_job_level_data)

### start collecting metrics
interval = int(config['GPU']['dmon_interval'])
is_collect_metric = config["INPUT_OUTPUT"]["collect_metric"]
### if scheduler need to collect metric for all jobs together
if is_collect_metric == "True" and not monitor_job_level_data:

    dmon_folder = folder + "/" + "all_job_nvidiasmidmon"
    if not os.path.exists(dmon_folder):
        os.makedirs(dmon_folder)
    file_name = scheduler_name + "_" + "dmon_data"
    collect_metric(deviceId, interval, dmon_folder, file_name)

## Start the scheduler
job_info = scheduler.start()

##Used when collecting for dat for all jobs together
if not monitor_job_level_data:
    kill_backgroud_process()
# TODO -- write all job info to data -- Dmon for each process (move to the sceduler)? or whole dataset?

# Write all the job info to the output file
if job_info:
    jobs_data = pd.DataFrame()
    for job in job_info:
        print(job.get_job_info())
        ## Get all the job attributes as a Series
        job_info = pd.Series(job.__dict__)
        jobs_data[job.name] = job_info

    jobs_data.to_csv(os.path.join(folder + "/jobs_data.csv"))
