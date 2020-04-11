#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# In order to execute a job, a thread is create to run the command in background.
# When the command finishes, it returns its output.
#
# Copyright © 2017 Marcelo Amaral <marcelo.amaral@bsc.es>

from __future__ import print_function

import copy
import subprocess
import threading
import time


def create_job(cmd, background=False, shell=False):
    e = Executor(cmd, background, shell)
    print("@Create Job: cmd: " + ''.join(cmd) + " Shell: " + shell.__str__() + " background: " + background.__str__())
    print("---Command recieved: " + ''.join(cmd))
    try:
        e.start()

    except:
        e.set_state(True)
    return e


class Executor(threading.Thread):
    def __init__(self, cmd, background=False, shell=False):
        threading.Thread.__init__(self)
        self.threadLock1 = threading.Lock()
        self.threadLock2 = threading.Lock()
        self.cmd = cmd
        self.background = background
        self.finished = False
        self.start_time = None
        self.finish_time = None
        self.shell = shell

    def execute_background(self, cmd):
        subprocess.Popen(cmd, shell=self.shell)
        self.set_finish_time(time.time())
        self.set_state(True)

    def execute(self, cmd):
        if self.shell is True:
            # popen = subprocess.Popen(cmd, shell=True)
            ## With check_call isntead of Popen, we can  join. As it is a "blocking operation"
            ##We wait untill our workload  finishes/ timeouts, we use this block operation  to join workload thread
            process = subprocess.check_call(cmd, shell=True)

        else:
            process = subprocess.check_call(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=self.shell)
            # popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell= self.shell)

        self.set_finish_time(time.time())
        self.set_state(True)
        return

    def run(self):
        self.set_start_time(time.time())
        if not self.background:

            workload = self.execute(self.cmd)

            try:
                while not self.has_finished():
                    out = workload.next()
                    if out is None:
                        break
            except:
                print("The process has finished")
        else:
            self.execute_background(self.cmd)

    def has_finished(self):
        self.threadLock1.acquire()
        state = copy.deepcopy(self.finished)
        self.threadLock1.release()
        return state

    def set_state(self, state):
        self.threadLock1.acquire()
        self.finished = state
        self.threadLock1.release()

    def get_time(self):
        self.threadLock2.acquire()
        s = copy.deepcopy(self.start_time)
        f = copy.deepcopy(self.finish_time)
        self.threadLock2.release()
        if f is None:
            f = time.time()
        return f - s

    def set_start_time(self, t):
        self.threadLock2.acquire()
        self.start_time = t
        self.threadLock2.release()

    def set_finish_time(self, t):
        self.threadLock2.acquire()
        self.finish_time = t
        self.threadLock2.release()
