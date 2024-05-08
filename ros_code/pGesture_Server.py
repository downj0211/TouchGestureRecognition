#! /usr/bin/env python3

import load_rospy
import rospy
import numpy as np
import torch
from std_msgs.msg import Float32MultiArray

from pGesture.pGesture_DataHandler import pGesture_DataHandler
from pGesture.pGesture_DataProcessor import pGesture_DataProcessor
from pGesture.pGesture_predict import pGestureNet

buffer_input = 100
wait_input   = 500
threshold    = 5


class pGesutre_Server:
    def __init__(self):
        rospy.init_node('pGesutre_Server')
        sub = rospy.Subscriber("/pHRI_teaching/gesture/data",Float32MultiArray, self.callback)

        # parameter initialize
        self.dh = pGesture_DataHandler(buffer_size = buffer_input,
                                 wait_size = wait_input)
        self.threshold = 5
        self.data = np.zeros(1)

        # set NN model
        self.net = pGestureNet().float()
        self.net.load_state_dict(torch.load('./pGestureNet_onlyG.pth'))

        pGesture_DataHandler(np.random.rand(100,22),"SAMPLE_REDUCTION",100)
        rospy.spin()

    def callback(self, data):
        data_input = np.reshape(np.array(data.data),(1,-1))

        # detect collision.
        F_norm = np.linalg.norm(data_input[:,9:11])

        if F_norm > self.threshold:
            self.dh.update_collisionflag(True)

        # predict gesture
        if self.dh.update_data(data_input):
            dataset = torch.tensor(self.dh.get_dataset()).float().unsqueeze(0).unsqueeze(0)
            outputs = self.net(dataset)
            _, predicted = torch.max(outputs.data, 1)

            print("[pGesture command] ",labels_map[predicted.item()],"\n")


if __name__ == "__main__":
    
    sub = pGesutre_Server()




