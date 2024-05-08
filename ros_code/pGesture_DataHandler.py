# pGesture_DataHandler.py
# class for handling data of pGesture

import math
import numpy as np

class pGesture_DataHandler:
    # initialize class
    def __init__(self, buffer_size = 100, wait_size = 500):
        self.buffer_size = buffer_size
        self.wait_size = wait_size

        self.init_parameters()

    # update data
    def update_data(self, input):
        # append data
        if self.first:
            self.data = input
            self.first = False
        else:
            self.data = np.append(self.data, input, axis = 0)

        # data shoud be lower than buffer when not collision
        if (self.data.shape[0] > self.buffer_size) and (not self.collision):
            self.data = np.delete(self.data, 0, axis = 0)

        self.wait_sample += 1

        # wait until satisfying the wait_size
        if (self.wait_sample > self.wait_size) and (self.collision):
            return True
        else:
            return False


    # update collision flag
    def update_collisionflag(self, iscollision):
        self.wait_sample = 0
        self.collision = iscollision

    # get command data
    def get_dataset(self):
        if (self.wait_sample > self.wait_size):
            data_out = self.data
            self.init_parameters()
            return data_out

        else:
            return np.zeros(1)

    # parameter initialize
    def init_parameters(self):
        self.data = np.zeros(1)
        self.wait_sample = 0
        self.collision = False
        self.first = True






