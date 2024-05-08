# pGesture_predict.py
# the class for predict pGesture command
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_labels import label_map


class touchgesture_model_CNN(nn.Module):
    def __init__(self, in_channel = 1, layer1 = 120, layer2 = 50,
                    pooling_size = 2, conv_size = 3, out_channel1 = 5, out_channel2 = 10,
                    M = 100, N = 46, label_type = 'GestureCommand'):
 
        super(touchgesture_model_CNN, self).__init__()

        self.input_dim = N
        self.conv1  = nn.Conv2d(in_channel, out_channel1, conv_size)
        self.pool   = nn.MaxPool2d(pooling_size, pooling_size)
        self.conv2  = nn.Conv2d(out_channel1, out_channel2, conv_size)

        self.mat_size = ((((M - (conv_size - 1))//2) - (conv_size - 1))//2)*((((N - (conv_size - 1))//2) - (conv_size - 1))//2)*out_channel2

        self.fc1    = nn.Linear(self.mat_size, layer1)
        self.fc2    = nn.Linear(layer1, layer2)
        self.fc3    = nn.Linear(layer2, len(label_map[label_type]))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.mat_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
