import os
import time

from src import nvmlgpu
from src import workload_executor
from src.scheduler import Job
from src.utils import utility

'''Default application Clock Scheduler
'''


class DefaultApplicationClockScheduler:
    def __init__(self, logger, application_que, workload_settings, deviceId, data_directory, folder,
                 default_application_clocks=None, montor_job_level_data=False):
        self.application_que = application_que
        self.logger = logger
        self.workload_settings = workload_settings
        self.deviceId = deviceId
        self.default_application_clocks = default_application_clocks
        self.data_directory = data_directory
        self.folder = folder
        self.run_type = "direct_run"  ## TODO - read from config / parameter
        self.montor_job_level_data = montor_job_level_data

    # Creates a list of Job Objects for Scheduler
    def get_job_list(self):
        job_list = []
        for application in self.application_que:
            job = Job.Job(self.workload_settings, application)
            job_list.append(job)

        return job_list

    def execute_job(self, job):
        workload_executor_service = workload_executor.WorkloadExecutor(self.logger, job.name, self.workload_settings,
                                                                       self.run_type, self.folder, "ss")
        workload_executor_service.start()

    def schedule_workload(self):
        job_list = self.get_job_list()
        ## Sort the jobs based on the job's deadline (arrivaltim+ exec_deadline)
        job_list = sorted(job_list, key=lambda x: x.sys_time_deadline, reverse=False)

        gpu_manager = nvmlgpu.GPUClockManagement(self.deviceId)
        start_time = time.time()

        for job in job_list:

            ## Get's the input feature list for our prediction model

            elapsed_time = start_time - time.time()
            ## if the job has arrived based on its distribution
            while job.arrival_time < elapsed_time:
                time.sleep(0.5)  ##wait until the  the job is avlaiable
                elapsed_time = start_time - time.time()

            ## if the job has arrived based on its distribution
            if job.arrival_time >= elapsed_time:

                job.mem_clock, job.sm_clock = self.default_application_clocks

                print("Job: {} Selected Application Clock: {} - {}".format(job.name, job.mem_clock, job.sm_clock))
                print("Job:{} Deadline: {} : Job  exec time:{}".format(job.name, job.exec_time_deadline, job.exec_time))

                gpu_manager.set_application_clocks(job.mem_clock, job.sm_clock, self.deviceId)
                print("Clocks are set, starting the application execution")

                if self.montor_job_level_data:

                    dmon_folder = self.folder + "/" + "qeaware_nvidiasmidmon"
                    if not os.path.exists(dmon_folder):
                        os.makedirs(dmon_folder)
                    utility.Utils().collect_metric(self.deviceId, 1, dmon_folder, job.name)

                    # This sleep function is to allow dmon thread to start and reach steady state before starting the application
                    time.sleep(2)
                    ### Workload Executor
                    job_start_time = time.time()

                    self.execute_job(job)
                    job.is_executed = True
                    # In simulation
                    # time.sleep(job.predicted_time)

                    job_end_time = str(time.time() - job_start_time)
                    job.scheduled_exec_time = job_end_time

                    self.logger.info("## Job " + str(job.name) + " has finished in: ")
                    ## sleep for 2 secs to allow dmon thread to reach steady state in its data collection . Also lowest record interval
                    # is 1 sec, which requires time to log the the data after workload execution
                    time.sleep(2)

                    # Kill metric collector thread

                    print("Starting application: {} kill".format(job.name))
                    utility.Utils().kill_backgroud_process()
                    ############
                ##collect dmon data for all jobs together
                else:
                    print("Clocks are set, starting the applicaiton execution")
                    job_start_time = time.time()

                    self.execute_job(job)

                    job.is_executed = True
                    # In simulation
                    # time.sleep(job.predicted_time)

                    job_end_time = str(time.time() - job_start_time)
                    job.scheduled_exec_time = job_end_time

                    self.logger.info("## Job " + str(job.name) + " has finished in: ")
        total_elapsed_time = str(time.time() - start_time)

        self.logger.info("## Executing all workload finished--> " + " Total time spent: " + total_elapsed_time)

        return job_list

    def start(self):
        self.logger.info("Scheduling Workload Started")
        job_info = self.schedule_workload()
        self.logger.info("Scheduling Workload Finished -- Returning to Main")

        return job_info
