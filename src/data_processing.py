import os
import sqlite3

import pandas as pd


class NvidiaSmiDmonDataProcessor:

    def __init__(self, file_name):
        self.file_name = file_name

    def get_average_data(self, preprocess_file):
        # self.file_name = "/home/shashi/phd_data/code/GPUETM_lillebranch/GPUETM/output/metrics" + "/" +  "temp" #"particlefilter2_2877_637"
        # ## filter the file with intermediate headers, that makes easy to load in the panda dataframe
        if preprocess_file:
            # Convert raw dmon output file (tab seperarted and multiple headers)  to standard CSV file.
            script_path = os.path.join(os.getcwd() + "/etc/scripts/filterscript.sh")
            cmd = "sh " + script_path + " " + self.file_name
            os.system(cmd)

        df = pd.read_csv(self.file_name, sep=",")
        df_mean = df.mean()
        return df_mean


class NVProfProfilerDataProcessor:

    def __init__(self, file_name):
        self.file_name = file_name

    def get_data(self):
        df = pd.read_csv(self.file_name, sep=",", skiprows=5)

        return df


##################NSight Data Parser
class NSightProfileDataProcessor:
    def __init__(self, sqlite_database, config=None, logger=None, application=None):
        self.config = config
        self.application = application
        self.logger = logger
        self.sqlite_database = sqlite_database
        ## following are the tables within the sqlite database
        self.CUPTI_ACTIVITY_KIND_KERNEL = None
        self.CUPTI_ACTIVITY_KIND_MEMCPY = None
        self.cudaAPIStats = None
        self.cudaKernelStats = None
        self.cudaMemoryOperationByteStats = None
        self.cudaMemoryOperationTimeStats = None
        self.CUPTI_ACTIVITY_KIND_MEMSET = None
        self.osrtAPIStats = None
        self.OSRT_CALLCHAINS = None
        ## Event ids exist, make sure how to use it
        self.OSRT_API = None
        self.CUPTI_ACTIVITY_KIND_RUNTIME = None
        ## Following tables  have categorical values with events or not useful data.
        self.MemcpyOperationStrings = None
        self.PROFILER_OVERHEAD = None
        self.StringIds = None
        self.UnwindMethodType = None

    def load_data(self):
        nsight_sqlite_data = self.sqlite_database
        conn = sqlite3.connect(nsight_sqlite_data)
        # load all the tables as panda's data frame
        # self.CUPTI_ACTIVITY_KIND_KERNEL = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL", conn)
        # cursor= conn.cursor()
        # cursor.execute("SELECT AVG (registersPerThread) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        #################data not exist  for myocyte.

        # self.CUPTI_ACTIVITY_KIND_KERNEL = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL", conn)
        # CREATE TABLE `CUPTI_ACTIVITY_KIND_KERNEL` (
        #     `id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        #     `start`	INT NOT NULL,
        #     `end`	INT NOT NULL,
        #     `deviceId`	INT NOT NULL,
        #     `contextId`	INT NOT NULL,
        #     `streamId`	INT NOT NULL,
        #     `correlationId`	INT NOT NULL,
        #     `eventClass`	INT NOT NULL,
        #     `globalPid`	INT NOT NULL,
        #     `demangledName`	INT NOT NULL,
        #     `shortName`	INT NOT NULL,
        #     `launchType`	INT NOT NULL,
        #     `cacheConfig`	INT NOT NULL,
        #     `registersPerThread`	INT NOT NULL,
        #     `gridX`	INT NOT NULL,
        #     `gridY`	INT NOT NULL,
        #     `gridZ`	INT NOT NULL,
        #     `blockX`	INT NOT NULL,
        #     `blockY`	INT NOT NULL,
        #     `blockZ`	INT NOT NULL,
        #     `staticSharedMemory`	INT NOT NULL,
        #     `dynamicSharedMemory`	INT NOT NULL,
        #     `localMemoryPerThread`	INT NOT NULL,
        #     `localMemoryTotal`	INT NOT NULL,
        #     `gridId`	INT NOT NULL,
        #     `sharedMemoryExecuted`	INT NOT NULL
        # );

        self.CUPTI_ACTIVITY_KIND_MEMCPY = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY", conn)
        # CREATE TABLE `CUPTI_ACTIVITY_KIND_MEMCPY` (
        # 	`id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        # 	`start`	INT NOT NULL,
        # 	`end`	INT NOT NULL,
        # 	`deviceId`	INT NOT NULL,
        # 	`contextId`	INT NOT NULL,
        # 	`streamId`	INT NOT NULL,
        # 	`correlationId`	INT NOT NULL,
        # 	`eventClass`	INT NOT NULL,
        # 	`globalPid`	INT NOT NULL,
        # 	`copyKind`	INT NOT NULL,
        # 	`srcKind`	INT NOT NULL,
        # 	`bytes`	INT NOT NULL
        # );

        self.cudaAPIStats = pd.read_sql_query("SELECT * FROM cudaAPIStats", conn)
        # CREATE TABLE `cudaAPIStats` (
        # 	`nameId`	INTEGER,
        # 	`num`	INTEGER,
        # 	`min`	INTEGER,
        # 	`max`	INTEGER,
        # 	`avg`	INTEGER,
        # 	`total`	INTEGER
        # );

        self.cudaKernelStats = pd.read_sql_query("SELECT * FROM cudaKernelStats", conn)
        # CREATE TABLE `cudaKernelStats` (
        # 	`shortName`	INTEGER,
        # 	`num`	INTEGER,
        # 	`min`	INTEGER,
        # 	`max`	INTEGER,
        # 	`avg`	INTEGER,
        # 	`total`	INTEGER
        # );
        self.cudaMemoryOperationByteStats = pd.read_sql_query("SELECT * FROM cudaMemoryOperationByteStats", conn)
        # CREATE TABLE `cudaMemoryOperationByteStats` (
        # 	`num`	INTEGER,
        # 	`min`	INTEGER,
        # 	`max`	INTEGER,
        # 	`avg`	INTEGER,
        # 	`total`	INTEGER,
        # 	`name`	TEXT
        # );
        self.cudaMemoryOperationTimeStats = pd.read_sql_query("SELECT * FROM cudaMemoryOperationTimeStats", conn)
        # CREATE TABLE `cudaMemoryOperationTimeStats` (
        # 	`num`	INTEGER,
        # 	`min`	INTEGER,
        # 	`max`	INTEGER,
        # 	`avg`	INTEGER,
        # 	`total`	INTEGER,
        # 	`name`	TEXT
        # );
        self.CUPTI_ACTIVITY_KIND_MEMSET = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET", conn)
        # CREATE TABLE `CUPTI_ACTIVITY_KIND_MEMSET` (
        # 	`id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        # 	`start`	INT NOT NULL,
        # 	`end`	INT NOT NULL,
        # 	`deviceId`	INT NOT NULL,
        # 	`contextId`	INT NOT NULL,
        # 	`streamId`	INT NOT NULL,
        # 	`correlationId`	INT NOT NULL,
        # 	`eventClass`	INT NOT NULL,
        # 	`globalPid`	INT NOT NULL,
        # 	`value`	INT NOT NULL,
        # 	`bytes`	INT NOT NULL
        # );
        self.CUPTI_ACTIVITY_KIND_RUNTIME = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME", conn)
        # CREATE TABLE `CUPTI_ACTIVITY_KIND_RUNTIME` (
        # 	`id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        # 	`start`	INT NOT NULL,
        # 	`end`	INT NOT NULL,
        # 	`eventClass`	INT NOT NULL,
        # 	`globalTid`	INT NOT NULL,
        # 	`correlationId`	INT NOT NULL,
        # 	`nameId`	INT NOT NULL,
        # 	`returnValue`	INT NOT NULL
        # );
        self.OSRT_API = pd.read_sql_query("SELECT * FROM OSRT_API", conn)
        # CREATE TABLE `OSRT_API` (
        # 	`id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        # 	`start`	INT NOT NULL,
        # 	`end`	INT NOT NULL,
        # 	`eventClass`	INT NOT NULL,
        # 	`globalTid`	INT NOT NULL,
        # 	`correlationId`	INT NOT NULL,
        # 	`nameId`	INT NOT NULL,
        # 	`returnValue`	INT NOT NULL,
        # 	`nestingLevel`	INT NOT NULL,
        # 	`callchainId`	INT NOT NULL
        # );
        self.osrtAPIStats = pd.read_sql_query("SELECT * FROM osrtAPIStats", conn)
        # CREATE TABLE `osrtAPIStats` (
        # 	`nameId`	INTEGER,
        # 	`num`	INTEGER,
        # 	`min`	INTEGER,
        # 	`max`	INTEGER,
        # 	`avg`	INTEGER,
        # 	`total`	INTEGER
        # );
        ############################events, ## not so useful data
        # -------------------------------------------------------------------------------------------------------------------------------------------------------#

        self.MemcpyOperationStrings = pd.read_sql_query("SELECT * FROM MemcpyOperationStrings", conn)
        # CREATE TABLE `MemcpyOperationStrings` (
        # 	`id`	INTEGER,
        # 	`name`	TEXT
        # );

        self.OSRT_CALLCHAINS = pd.read_sql_query("SELECT * FROM OSRT_CALLCHAINS", conn)
        # CREATE TABLE `OSRT_CALLCHAINS` (
        # 	`id`	INT NOT NULL,
        # 	`symbol`	INT NOT NULL,
        # 	`module`	INT NOT NULL,
        # 	`kernelMode`	INT NOT NULL,
        # 	`thumbCode`	INT NOT NULL,
        # 	`unresolved`	INT NOT NULL,
        # 	`specialEntry`	INT NOT NULL,
        # 	`originalIP`	INT NOT NULL,
        # 	`unwindMethod`	INT NOT NULL,
        # 	`stackDepth`	INT NOT NULL,
        # 	PRIMARY KEY(`id`,`stackDepth`)
        # );
        self.PROFILER_OVERHEAD = pd.read_sql_query("SELECT * FROM PROFILER_OVERHEAD", conn)
        # CREATE TABLE `PROFILER_OVERHEAD` (
        # 	`id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        # 	`start`	INT NOT NULL,
        # 	`end`	INT NOT NULL,
        # 	`eventClass`	INT NOT NULL,
        # 	`globalTid`	INT NOT NULL,
        # 	`correlationId`	INT NOT NULL,
        # 	`nameId`	INT NOT NULL,
        # 	`returnValue`	INT NOT NULL
        # );
        self.StringIds = pd.read_sql_query("SELECT * FROM StringIds", conn)
        # CREATE TABLE `StringIds` (
        # 	`id`	INTEGER,
        # 	`value`	TEXT NOT NULL,
        # 	PRIMARY KEY(`id`)
        # );
        self.UnwindMethodType = pd.read_sql_query("SELECT * FROM UnwindMethodType", conn)
        # CREATE TABLE `UnwindMethodType` (
        # 	`number`	INTEGER,
        # 	`name`	TEXT NOT NULL,
        # 	PRIMARY KEY(`number`)
        # );
        self.ThreadNames = pd.read_sql_query("SELECT * FROM ThreadNames", conn)
        # CREATE TABLE `ThreadNames` (
        #     `id`	INTEGER PRIMARY KEY AUTOINCREMENT,
        #     `nameId`	INT NOT NULL,
        #     `priority`	INT NOT NULL,
        #     `globalTid`	INT NOT NULL
        # );
        self.UnwindMethodType = pd.read_sql_query("SELECT * FROM UnwindMethodType", conn)
        # CREATE TABLE `UnwindMethodType` (
        # 	`number`	INTEGER,
        # 	`name`	TEXT NOT NULL,
        # 	PRIMARY KEY(`number`)
        # );
        conn.close()

    def get_averaged_data_from_tables(self):
        print("Processing the Nsight Data for file: {}".format(self.sqlite_database))
        self.load_data()
        #  data = self.CUPTI_ACTIVITY_KIND_KERNEL[['launchType', 'cacheConfig', 'registersPerThread', 'gridX',
        # 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'staticSharedMemory',
        # 'dynamicSharedMemory', 'localMemoryPerThread', 'localMemoryTotal',
        # 'gridId', 'sharedMemoryExecuted']]
        #  data= data.mean()

        data = data['CUPTI_ACTIVITY_KIND_MEMCPY_byte'] = self.CUPTI_ACTIVITY_KIND_MEMCPY['bytes'].mean()

        data['CUPTI_ACTIVITY_KIND_MEMSET_bytes'] = self.CUPTI_ACTIVITY_KIND_MEMSET['bytes'].mean()

        data['OSRT_CALLCHAINS_stackDepth'] = self.OSRT_CALLCHAINS['stackDepth'].mean()

        data['cudaAPIStats_total'] = self.cudaAPIStats['total'].mean()
        data['cudaKernelStats_total'] = self.cudaKernelStats['total'].mean()
        data['cudaMemoryOperationByteStats_total'] = self.cudaMemoryOperationByteStats['total'].mean()
        data['cudaMemoryOperationTimeStats_total'] = self.cudaMemoryOperationTimeStats['total'].mean()
        data['osrtAPIStats_total'] = self.osrtAPIStats['total'].mean()

        return data

#
# nsight_sqlite_database = "/home/shashi/Dropbox/report3.sql ite"
# profiler_data= NSightProfilerData(nsight_sqlite_database)
# profiler_data.load_data()
# average_data = profiler_data.get_averaged_data_from_tables()
# average_data.to_csv("checkcsv.csv", header= False)
