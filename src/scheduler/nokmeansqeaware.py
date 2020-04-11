import os
import time

import joblib
import numpy as np
import pandas as pd

from src import nvmlgpu
from src import workload_executor
from src.scheduler import Job
from src.utils import utility

'''QoSandEnergy-aware-DataDriven-DVFS-Scheduler (EPDVFSS)
class. This class implements the '''


class QEDVFSScheduler:
    def __init__(self, logger, application_que, workload_settings, deviceId, data_directory, folder,
                 monitor_job_level_data=None):
        self.application_que = application_que
        self.logger = logger
        self.workload_settings = workload_settings
        self.deviceId = deviceId
        #        self.default_application_clocks = default_application_clocks
        self.data_directory = data_directory
        self.folder = folder
        self.run_type = "direct_run"  ## TODO - read from config / parameter
        self.monitor_job_level_data = monitor_job_level_data

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

    ## For the input job list,  get the  feature list from its defualt application clock profile
    def get_feature_set(self, job_name):

        # df= pd.read_csv(os.path.join(self.data_directory + "default_clock_data.csv"))
        # df = df[(df["Application_name"] == job_name)]
        print("Job Feature Set: --------" + job_name)
        df = pd.read_csv(
            os.path.join(self.data_directory + "combined_metrics/" + job_name + "_dmon_nvprof_combined.csv"))
        ##for xgboost
        # independent_variables = open("src/prediction_models/filtered_independent_variables.txt").read().splitlines()
        independent_variables = open("src/prediction_models/filtered_independent_variables.txt").read().splitlines()
        categorical_features = open("src/prediction_models/excluded_categorical_columns.txt").read().splitlines()
        independent_variables += categorical_features

        ##########Convert existing categorical to numerical
        for header in categorical_features:
            df[header] = df[header].astype('category').cat.codes
            df[header] = df[header].astype('category').cat.codes

        feature_set = pd.DataFrame(df, columns=independent_variables)
        ## Replace the special characters/unit names and convert dataframe to float type
        feature_set[feature_set.columns] = feature_set[feature_set.columns].replace(
            {'%': '', ',': '', '/s': '', 'GB': '', 'MB': '', 'B': '', 'K': ''}, regex=True)
        #    feature_set = feature_set.astype(float)

        return feature_set

    ## For given application and for all clock set, predict the power and time
    def predict_power_time(self, name, pwr_prediction_model, time_prediction_model):

        prediction_result = dict()
        try:

            feature_set = self.get_feature_set(name)

            pwr_model = joblib.load(pwr_prediction_model)
            time_model = joblib.load(time_prediction_model)
            self.logger.info("Models are Loaded: {}-{} ".format(pwr_prediction_model, time_prediction_model))

            ## For different clock set predict the power, TeslaP100 and V100 have only one mem_clock
            mem_clock = feature_set['mem_clock_x'].unique()
            mem_clock = mem_clock[0]
            sm_clock_list = feature_set['sm_clock'].unique()

            for sm_clock in sm_clock_list:
                ## get input vector for the current clock set
                prediction_input = feature_set[
                    (feature_set['mem_clock_x'] == mem_clock) & (feature_set['sm_clock'] == sm_clock)]

                # taking average of kernel values
                # prediction_input = pd.DataFrame(prediction_input.mean()).transpose()

                predicted_pwr = pwr_model.predict(prediction_input)
                predicted_time = time_model.predict(prediction_input)

                print ("$$$$$$$$$$$$$$$$$$$$$$$$Precited pwr {}".format(predicted_pwr))
                print ("^^^^^^^^^^^^^^^precited time {}".format(predicted_time))

                print(
                    'model predicted for clock {}_{} power: {} : time: {}'.format(mem_clock, sm_clock, predicted_pwr[0],
                                                                                  predicted_time[0]))
                self.logger.info(
                    'model predicted for clock {}_{} power: {}'.format(mem_clock, sm_clock, predicted_pwr[0],
                                                                       predicted_time[0]))
                prediction_result['{}_{}'.format(mem_clock, sm_clock)] = np.array(
                    [mem_clock, sm_clock, predicted_pwr[0], predicted_time[0]])


        except Exception as e:
            print('Exeption in prediction: {}'.format(e.__str__()))
            self.logger.error("No model here, check the path'")

        return prediction_result

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
            if job.arrival_time >= elapsed_time:

                ## Predict the Power for
                ## TODO configure prediction model used
                dir = self.data_directory + "prediction_models/" + "catboost"
                print("prediciton model data directory:{}".format(self.data_directory))
                pwr_prediction_model = dir + "/" + job.name + "_pwr" + ".pkl"
                time_prediction_model = dir + "/" + job.name + "_time" + ".pkl"

                time_for_prediction_start = time.time()
                prediction_result = self.predict_power_time(job.name, pwr_prediction_model, time_prediction_model)
                job.time_for_prediction = time_for_prediction_start - time.time()

                # get the near-optimal clock set from predicted result -- > Clock set that meet the deadline and consumes less power based on prediction result
                min_power = 999999
                for prediction in prediction_result.values():
                    mem_clock, sm_clock, predicted_pwr, predicted_time = prediction
                    # TODO add kmeans and alpha paramter for the deadline
                    ##alpha paramter acts as relaxation paramter for the deadline, overstimating the deadline to adjust the system/ prediction error margins
                    alpha = 0.1
                    deadline = job.exec_time_deadline + job.exec_time_deadline * alpha
                    if predicted_time < deadline and predicted_pwr < min_power:
                        min_power = predicted_pwr
                        job.mem_clock = int(mem_clock)
                        job.sm_clock = int(sm_clock)
                        job.predicted_power = predicted_pwr
                        job.predicted_time = predicted_time

                print("Job: {} Selected Application Clock: {} - {}".format(job.name, job.mem_clock, job.sm_clock))
                print(
                    "Job:{} Predicted Power  {}- PrecitedTime for clock: {} : Deadline: {} : Job  exec time:{}".format(
                        job.name, job.predicted_power, job.predicted_time,
                        job.exec_time_deadline, job.exec_time))

                gpu_manager.set_application_clocks(job.mem_clock, job.sm_clock, self.deviceId)

                if self.monitor_job_level_data:
                    print (
                        "IN FIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")

                    dmon_folder = self.folder + "/" + "qeaware_nvidiasmidmon"
                    if not os.path.exists(dmon_folder):
                        os.makedirs(dmon_folder)
                    utility.Utils().collect_metric(self.deviceId, 1, dmon_folder, job.name)
                    print (
                        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

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
                    print (
                        "IN ELSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                    print (self.monitor_job_level_data)
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
