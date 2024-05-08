#! /usr/bin/env python3

import sys
import os
from pGesture import load_rospy
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray

from pGesture.pGesture_DataHandler import pGesture_DataHandler
from pGesture.pGesture_predict import pGestureNet

# setting parameters
data_directory = "experiment_0506"
buffer_input = 100
wait_input   = 500
threshold    = 5

class pGesture_Learning_Server:
    def __init__(self, gesture_type, contact_link):
        rospy.init_node('pGesture_Learning_Server')
        sub = rospy.Subscriber("/pHRI_teaching/gesture/data",Float32MultiArray, self.callback)

        # parameter initialize
        self.dh = pGesture_DataHandler(buffer_size = buffer_input,
                                 wait_size = wait_input)
        self.threshold = threshold
        self.data = np.zeros(1)
        self.gesture_type = gesture_type
        self.contact_link = contact_link

        # save directory initialize
        self.file_path = os.path.join('/home/ubuntu/down_ws/NNdata',data_directory, gesture_type)

        if not os.path.exists(self.file_path):
            print("[Error] There is no directory to save the physical gesture data\n(", self.file_path,")")
            quit()

        self.file_num = 1
        while os.path.exists(os.path.join(self.file_path, self.gesture_type+'_link'+str(self.contact_link)+'_'+str(self.file_num)+'.csv')):
            self.file_num += 1

        print('[',gesture_type, 'link', str(contact_link),'] ', str(self.file_num - 1), 'file exist')
        print('Ready to get the physical gesture data set!!')

        rospy.spin()

    def callback(self, data):
        data_input = np.reshape(np.array(data.data),(1,-1))

        # detect collision
        F_norm = np.linalg.norm(data_input[:,9:11])

        if F_norm > self.threshold:
            self.dh.update_collisionflag(True)

        # get data matrix
        if self.dh.update_data(data_input):
            dataset = self.dh.get_dataset()

            filename = os.path.join(self.file_path, self.gesture_type+'_link'+str( self.contact_link)+'_'+str(self.file_num)+'.csv')
            np.savetxt(filename, dataset, delimiter = ',')

            print('[', self.gesture_type, 'link', str(self.contact_link),'] ', str(self.file_num), '-th file save')
            self.file_num += 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("[Error] Not enough input parameters!!\n")
        quit()

    sub = pGesture_Learning_Server(gesture_type = sys.argv[1], contact_link = sys.argv[2])




