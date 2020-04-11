class Job:
    def __init__(self, workload_settings, application):
        # Loading the job's resources
        self.name = workload_settings[application]['app_name']
        self.exec_time = workload_settings[application]['execution_time']
        self.arrival_time = workload_settings[application]['arrival_time']
        self.exec_time_deadline = workload_settings[application]['exec_time_deadline']  ##absolute value
        self.sys_time_deadline = workload_settings[application]['sys_time_deadline']  ## arrival+ exec_time_deadline
        self.job_path = workload_settings[application]['path']
        self.runfile = workload_settings[application]['runfile']
        self.mem_clock = None
        self.sm_clock = None
        self.predicted_time = None
        self.predicted_power = None
        self.scheduled_exec_time = None
        self.is_executed = False
        self.time_for_prediction = None

    def get_job_name(self):
        return self.name

    def get_exec_time(self):
        return self.exec_time

    def get_deadline(self):
        return self.deadline

    def get_job_path(self):
        return self.job_path

    def get_runfile(self):
        return self.runfile

    def set_mem_clock_predicted(self, mem_clock):
        self.mem_clock_predicted = mem_clock

    def set_sm_clock_predicted(self, sm_clock):
        self.sm_clock_predicted = sm_clock

    ## Print the Job Information --> If the Job is executed, it prints  its execution history and general information
    ## Or else just prints general info and error message
    def get_job_info(self):

        info = "Job: " + str(self.name)
        if self.is_executed:
            info += " Executed with clock set: {}_{}".format(self.mem_clock, self.sm_clock)
            info += " Application exec_time: {} : deadline: {}".format(self.exec_time, self.exec_time_deadline)
            info += " Actial Scheduled exec_time: {} and predicted exec_time: {}".format(self.scheduled_exec_time,
                                                                                         self.predicted_time)
            info += " and predicted power: {}".format(self.predicted_power)

        else:
            info += "did not execute"
        return info

    # def get_job_info_as_array():
