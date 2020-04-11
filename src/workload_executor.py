import os

from src.utils import executor


class WorkloadExecutor:

    def __init__(self, logger, application, workload_settings, run_type, folder=None, current_clock_set=None):
        self.application = application
        self.logger = logger
        self.workload_settings = workload_settings
        self.folder = folder
        self.current_clock_set = current_clock_set
        self.run_type = run_type

    def get_application_command(self):

        workload_path = os.path.join(
            os.getcwd() + "/workload/benchmark_applications" + self.workload_settings[self.application]['path'])
        # cmd = "cd "  + self.workload_settings[self.application]['path']  + " && "
        cmd = "cd " + workload_path + " && "

        if self.run_type == "nsight_profiler":
            # nsys should be system wide accessible or in the system path. Else, provide absolute path to its executable
            # "/home/silager/nsight-systems-2019.5.1/bin/nsys"
            profile_data_file = self.folder + "/" + self.application + "_" + self.run_type + "_" + self.current_clock_set
            cmd = cmd + "nsys  profile --stats=true --trace=cuda,cudnn,cublas,osrt,nvtx --delay=1  " + " -o " + profile_data_file + " --force-overwrite=true " + " python " + \
                  self.workload_settings[self.application]['runfile']

        elif self.run_type == "nvprof_profiler":
            ## %p is required for log file in nvprof when  child process traces are enabled, profiler data logged into file based on process ids
            profile_data_file = self.folder + "/" + self.application + "_" + self.run_type + "_processid-%p_" + self.current_clock_set
            cmd = cmd + "nvprof  --profile-child-processes  --metrics all  --csv  --log-file " + profile_data_file + " python " + \
                  self.workload_settings[self.application]['runfile']

        # When profiling is not required, application is run without any profiler and only the nvidia-smi dmon data is collected
        elif self.run_type == "direct_run":
            cmd = cmd + " python " + self.workload_settings[self.application]['runfile']

        return cmd

    def run_application(self):

        cmd = self.get_application_command()
        self.logger.info("Running: {} application".format(self.application)
                         + " cmd-> " + cmd)
        try:
            e = executor.create_job(cmd, background=False, shell=True)

        except:
            self.logger.error("Execution of the command failed, cmd -> " + cmd)
        return e

    def start(self):
        e = self.run_application()
        print("started Join for workload thread")
        e.join()
        print("finished join for workload thread")
