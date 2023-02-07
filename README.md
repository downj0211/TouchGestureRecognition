# TouchGestureRecognition

## Introduction

This project implements touch gesture recognition, which is the gesture of physical interaction by a human operator in a collaborative task, using Python 3.8. In this project, a collaborative robot, Franka Emika Panda, and its dynamic response dataset in the directory 'data' were used. (This research was submitted to IEEE transactions on cognitive and developmental systems as "Touch gesture-based physical human-robot interaction for collaborative task". 2022.11.29) 

** The used dataset in the proposed article will be uploaded later.

**Platform** : Python 3.8

**Libraries** : Pytorch, Numpy


## Usage

Run 'training.py' to begin the program.

First, define the deep learning model(among RNN, CNN, LSTM, GRU), its hyperparameters, criterion, and optimizer. Then, execute the function 'training_networks' with the defined information. The results of the function are 1)the trained touch gesture model in the directory 'learned_model' and 2)the log of training in the directory 'learned_model/log'.

If you want to use the trained model to predict the touch gesture for the real collaborative robot(Franka Emika Pand), move the codes in the directory 'ros_code' to the target ROS package of robot control. To obtain the dynamic pattern of robot, the dynamic responses(time, energy, momentum, wrench, and joint torques) should be published through the ROS Float32MultiArray topic as '/touch_gesture/data'. Then, the touch gesture will be recognized in real-time by executing 'TouchGestureServer.py'.
