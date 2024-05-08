#%%
# load dataloader
#%env CUDA_LAUNCH_BLOCKING=1
"""
Created on Wed Nov  9 12:16:46 2022

@author: down
"""

import torch
from scripts.pGesture_Dataset import pGesture_Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import models
from model_labels import label_map


transform = transforms.Compose([transforms.ToTensor()])
label_maps = label_map["GestureCommand"] 

input_dim = 46
hidden_dim = 512
layer_dim = 5
output_dim = len(label_maps)

#%%
#model_name = "model_CNN_221108"
#net = models['CNN']().to("cuda")

#model_name = "model_GRU_221108"
#net = models['GRU'](input_dim, hidden_dim, layer_dim, output_dim).to("cuda")

#model_name = "model_RNN_221108"
#net = models['RNN'](input_dim, hidden_dim, layer_dim, output_dim).to("cuda")

model_name = "model_LSTM_221108"
net = models['LSTM'](input_dim, hidden_dim, layer_dim, output_dim).to("cuda")


net.load_state_dict(torch.load('./learned_model/'+model_name+'.pth'))
net.eval()


test_dataset = pGesture_Dataset("annotations/annotations_testing.csv","./data/testing_data", 
                                 transform = transform)
f = open('validation/result('+model_name+').txt', 'w')

for idx in range(len(test_dataset)):
    dataset = test_dataset[idx]
    inputs = dataset['dataset'].reshape(-1, 100, input_dim).float().to("cuda")
    labels = dataset['label']
    
    outputs = net(inputs.float())
    predicted = F.log_softmax(outputs, dim=1).argmax(dim=1)
    
    f.write(str(labels) + '\t')
    f.write(str(predicted.item()) + '\t')
    if labels == predicted.item():
        f.write('1\t')
    else:
        f.write('0\t')
        
    if (labels//7) == (predicted.item()//7):
        f.write('1\t')
    else:
        f.write('0\t')
        
    if (labels%7) == (predicted.item()%7):
        f.write('1\n')
    else:
        f.write('0\n')
        
f.close()