import os

import joblib

from src.utils import executor


class Utils:

    # def __init__(self, config):
    #     self.config = config

    def dumpmodel(self, model, model_dir, filename):

        print("--> Dumping  ", model, " model to a persistent file")

        if not os.path.exists(model_dir):
            # os.mkdir(model_dir)
            os.makedirs(model_dir)
            print("--> Directory: ", model_dir, " Created")
        else:
            print("--> Directory ", model_dir, " already exists")

        filepath = model_dir + filename

        # Dump model using joblib
        joblib.dump(model, filepath)

        print("--> Model ", model, " dumped successfully to directory ", filepath)

    def collect_metric(self, deviceId, interval, output_folder, metric_file_name):
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

    def collect_metric2(self, interval, output_folder, metric_file_name):
        """This function creates a thread that will execute the script to collect system metrics"""
        cwd = os.getcwd()
        script_path = os.path.join(cwd + "etc/scripts/collect_gpu_metrics.sh")
        out_directory = output_folder
        cmd = list()
        cmd.append(script_path)
        cmd.append(str(interval))  # Monitoring Interval
        cmd.append(out_directory)  # Directory for the output
        cmd.append(metric_file_name)  # File name prefix for the experiment
        print("Collecting metrics , folder " + output_folder + " file- " + metric_file_name)
        executor.create_job(cmd, background=True)
        print("Collecting metrics finished")

    def kill_backgroud_process(self):
        """Kill the running backgound process"""
        cwd = os.getcwd()
        path = os.path.join(cwd, "etc/scripts/kill_processes.sh")
        # out_directory = output_folder + "/logs/"
        cmd = list()
        cmd.append(path)
        print("killing background processes")
        executor.create_job(cmd, background=True)
        print("killing background processes finished")
