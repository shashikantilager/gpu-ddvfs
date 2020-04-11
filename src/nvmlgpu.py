import subprocess

from py3nvml.py3nvml import *


#
# def check():
#     gpumanagement = GPUClockManagement()
#     gpumanagement.printClockInfo()


class GPUClockManagement:

    def __init__(self, deviceId):
        nvmlInit()
        self.deviceId = deviceId
        self.devicecount = nvmlDeviceGetCount()
        self.handle = nvmlDeviceGetHandleByIndex(deviceId)

    def print_clock_info(self):
        print("Device count: {} ".format(self.devicecount))
        print("Device {}: {}".format(0, nvmlDeviceGetName(self.handle)))
        # Info

        print("ClockInfo NVML_CLOCK_Graphics: {}".format(nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_GRAPHICS)))
        print("ClockInfo NVML_CLOCK_SM: {}".format(nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM)))
        print("ClockInfo NVML_CLOCK_Memory: {}".format(nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_MEM)))

        #### Application clock
        print("Application NVML_CLOCK_Graphics: {}".format(
            nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_GRAPHICS)))
        ## There is no application clock SM-- verify it
        print("Application NVML_CLOCK_SM: {}".format(nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_SM)))
        print("Application NVML_CLOCK_Memory: {}".format(nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_MEM)))

        #### Max Application Clock
        print("Max Application NVML_CLOCK_Graphics: {}".format(
            nvmlDeviceGetMaxClockInfo(self.handle, NVML_CLOCK_GRAPHICS)))
        ## There exist default application clock SM-- verify it
        print("Max Application NVML_CLOCK_SM: {}".format(nvmlDeviceGetMaxClockInfo(self.handle, NVML_CLOCK_SM)))
        print("Max  Application NVML_CLOCK_Memory: {}".format(nvmlDeviceGetMaxClockInfo(self.handle, NVML_CLOCK_MEM)))

        ##### Deafalut Clock
        print("Default Application NVM L_CLOCK_Graphics: {}".format(
            nvmlDeviceGetDefaultApplicationsClock(self.handle, NVML_CLOCK_GRAPHICS)))
        ## There is no max application clock SM-- verify it
        print("Default  Application NVML_CLOCK_SM: {}".format(
            nvmlDeviceGetDefaultApplicationsClock(self.handle, NVML_CLOCK_SM)))
        print("Default  Application NVML_CLOCK_Memory: {}".format(
            nvmlDeviceGetDefaultApplicationsClock(self.handle, NVML_CLOCK_MEM)))

        ### Autoboost
        # boostedState, boostedDefaultState = nvmlDeviceGetAutoBoostedClocksEnabled(self.handle)
        #
        # print(boostedDefaultState, boostedState)
        #### supported clocks
        print("Supported Memory Clocks: {}".format(nvmlDeviceGetSupportedMemoryClocks(self.handle)))
        # # [900, 405]
        # #
        # print("Supported Graphic Clocks Mem=900: {}".format(nvmlDeviceGetSupportedGraphicsClocks(self.handle, memoryClockMHz=900)))
        # # [1124, 1058, 1006]
        # print("Supported Graphic Clocks Mem=405: {}".format(nvmlDeviceGetSupportedGraphicsClocks(self.handle, memoryClockMHz=405)))
        # # [405]

        #### Throttle reasons
        print("Supported Clocks Throttle Reason: {}".format(nvmlDeviceGetSupportedClocksThrottleReasons(self.handle)))

    ###set clocks

    def get_application_clocks(self):
        memory_clock = nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_MEM)
        graphics_clock = nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_SM)
        print("Application NVML_CLOCK_SM: {}, NVML_CLOCK_Memory: {}".format(graphics_clock, memory_clock))

        return memory_clock, graphics_clock

    def set_application_clocks(self, mem_clock_mhz, graphics_clock_mhz, deviceId):
        print ("setting application clocks ---- ")
        ## current mem clocks 900, suported [405, 900]
        ## with 900 -- > supported graphic clocks are [1124, 1058, 1006]
        ##with 405 ---> supported graphic clocks are [405]

        # This funciton is not working as requires, changing to default command line tool
        # nvmlDeviceSetApplicationsClocks(self.handle, mem_clock_mhz, graphics_clock_mhz)
        # if didnt work, use follwing disables the autoboost
        # sudo    nvidia-smi -pm 1
        ##sudo    nvidia-smi --auto-boost-default = 0
        # #################################new code
        # cwd = os.getcwd()

        # path = "/home/shashi/phd_data/code/GPUETM/etc/scripts/set_clock.sh"
        path = os.path.join(os.getcwd() + "/etc/scripts/set_clock.sh")
        print(path)
        # out_directory = output_folder + "/logs/"
        # cmd = list()
        # cmd.append(path)
        # cmd.append(str(mem_clock_mhz))
        # cmd.append(str(graphics_clock_mhz))
        cmd = path + " " + str(deviceId) + " " + str(mem_clock_mhz) + " " + str(graphics_clock_mhz)
        # # print(''.join(cmd))
        subprocess.check_call(cmd, shell=True)

        # subprocess.call(['/home/shashi/phd_data/code/GPUETM/etc/scripts/set_clock.sh', str(mem_clock_mhz), str(graphics_clock_mhz)], shell=True)

        # subprocess.check_call(cmd, shell = True)
        # cmd.append(out_directory)  # Directory for the output
        # subprocess.Popen(cmd)

        # ####################
        # cmd = "sudo  nvidia-smi -i 0 -ac " + str(mem_clock_mhz) + ","  + str(graphics_clock_mhz)
        # os.system(cmd)
        # subprocess.check_call(cmd, shell= True)

        print ("Application Clocks are set to Mem: {} Graphics: {}".format(mem_clock_mhz, graphics_clock_mhz))
        # print("Application Clocks after setting Graphics:{}  SM:{} Mem:{}".format(nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_GRAPHICS),
        #                                                                nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_SM),
        #                                                                nvmlDeviceGetApplicationsClock(self.handle,NVML_CLOCK_MEM)))

    ###reset -- default Graphics:1058  SM:1058 Mem:900
    def reset_application_clocks(self):
        nvmlDeviceResetApplicationsClocks(self.handle)
        print("Reset is done")
        print("Application Clocks after resettingclocks  Graphics:{}  SM:{} Mem:{}".format(
            nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_GRAPHICS),
            nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_SM),
            nvmlDeviceGetApplicationsClock(self.handle, NVML_CLOCK_MEM)))

    ## Autoboost disables automatic scaling of clocks based on power
    ## This function is not working properly
    ## use default command sudo nvidia-smi --auto-boost-default=0
    def disbaleAutoBoostedClock(self):
        nvmlDeviceSetAutoBoostedClocksEnabled(self.handle, enabled=0)
        print("AutoBoostedClock disabled:  ")

    def enableAutoBoostedClock(self):
        nvmlDeviceSetAutoBoostedClocksEnabled(self.handle, enabled=1)
        print("AutoBoostedClock Enabled: Status: ")

    # ##powermanagement
    # powMan = nvmlDeviceGetPowerManagementMode(self.handle)
    # print("PowErManagement Mode 1 supported 0 not supported-> Value: {}".format(powMan) )
    #

    def start(self):
        self.print_clock_info()

# #########TEST
# s= GPUClockManagement(0)
# # #
# s.print_clock_info()
# s.get_application_clocks()
# # s.set_application_clocks(mem_clock_mhz=405, graphics_clock_mhz=405)
# s.set_application_clocks(mem_clock_mhz=900, graphics_clock_mhz=1124)
# # s.get_application_clocks()

# class NvidiaSmiDmonDataProcessor:
#
#     def __init__(self, file_name):
#         self.file_name = file_name
#
#     def get_average_data(self):
#         # self.file_name = "/home/shashi/phd_data/code/GPUETM_lillebranch/GPUETM/output/metrics" + "/" +  "temp" #"particlefilter2_2877_637"
#     # ## filter the file with intermediate headers, that makes easy to load in the panda dataframe
#         script_path = os.path.join(os.getcwd()  + "/etc/scripts/filterscript.sh")
#         # cmd =   "sh "+ "/home/shashi/phd_data/code/GPUETM_lillebranch/GPUETM/etc/scripts/filterscript.sh " + self.file_name
#         cmd = "sh " + script_path +  " "  +self.file_name
#         # cmd=   "sh "+ os.getcwd() + "/" + "etc/scripts/filterscript.sh " + file_name
#         os.system(cmd)
#         df = pd.read_csv(self.file_name, sep= ",")
#         df_mean = df.mean()
#         return df_mean
