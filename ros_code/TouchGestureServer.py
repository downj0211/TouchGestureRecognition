#! /usr/bin/env python3

import load_rospy
import rospy
import numpy as np
import torch
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
import pandas as pd

from script.TouchGestureDynamicsHandler import TouchGestureDynamicsHandler
from script.TouchGesturePreprocessing import TouchGesturePreprocessing
from models import models
from model_labels import label_map

buffer_input = 100
wait_input   = 500
threshold    = 5
label_maps = label_map["GestureCommand"]

class TouchGestureServer:
    def __init__(self):
        rospy.init_node('TouchGestureServer')
        sub = rospy.Subscriber("/touch_gesture/data",Float32MultiArray, self.callback)

        # parameter initialize
        self.dh = TouchGestureDynamicsHandler(buffer_size = buffer_input,
                                 wait_size = wait_input)
        self.threshold = 5
        self.data = np.zeros(1)

        # set NN model
        input_dim = 46
        hidden_dim = 512
        layer_dim = 5
        output_dim = len(label_map['GestureCommand'])
        
        self.net = models['GRU'](input_dim, hidden_dim, layer_dim, output_dim)
        self.net.load_state_dict(torch.load('./learned_model/model_GRU_221108_2.pth'))

        TouchGestureDynamicsHandler(np.random.rand(100,22),"SAMPLE_REDUCTION",100)
        rospy.spin()

    def callback(self, data):
        data_input = np.reshape(np.array(data.data),(1,-1))

        # detect collision.
        F_norm = np.linalg.norm(data_input[:,9:11])

        if F_norm > self.threshold:
            self.dh.update_collisionflag(True)

        # predict gesture
        if self.dh.update_data(data_input):
#                self.state = 0
            data = self.dh.get_dataset()
            dataset = pGesture_DataProcessor(data = data,
                                            SAMPLE_REDUCTION = 100,
                                            HIGHPASS = 30,
                                            NORMALIZATION = True)

            outputs = self.net(torch.Tensor(dataset).unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            print("[Gesture Recognized] ",label_maps[predicted.item()])

            pub = rospy.Publisher("/touch_gesture/predicted", Int32, queue_size=10)
            sleep(0.1)
            pub.publish(predicted.item())

            print("[Gesture Command] ",labels_map[predicted.item()],"\n")


if __name__ == "__main__":
    
    sub = TouchGestureServer()




