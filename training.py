#%%
# load dataloader
#%env CUDA_LAUNCH_BLOCKING=1
import torch.nn as nn
import torch.optim as optim
from model_labels import label_map
from models import models
from training_networks import training_networks

#%%
input_dim = 46
hidden_dim = 512
layer_dim = 5
output_dim = len(label_map['GestureCommand'])

lr = 0.001
n_epochs = 190
    
### criterion setting
criterion = nn.CrossEntropyLoss() #standard code

#%%
print('LSTM network training start!!')
net_LSTM = models['LSTM'](input_dim, hidden_dim, layer_dim, output_dim).to("mps")
optimizer_LSTM = optim.Adam(net_LSTM.parameters())

training_networks(net = net_LSTM, criterion = criterion, optimizer = optimizer_LSTM, 
                  lr = lr, epochs = n_epochs, filename = "LSTM_221108")

#%%
print('RNN network training start!!')
net_RNN = models['RNN'](input_dim, hidden_dim, layer_dim, output_dim).to("mps")
optimizer_RNN = optim.Adam(net_RNN.parameters())

training_networks(net = net_RNN, criterion = criterion, optimizer = optimizer_RNN, 
                  lr = lr, epochs = n_epochs, filename = "RNN_221108")

#%%
print('CNN network training start!!')
net_CNN = models['CNN']().to("mps")
optimizer_CNN = optim.Adam(net_CNN.parameters())

training_networks(net = net_CNN, criterion = criterion, optimizer = optimizer_CNN, 
                  lr = lr, epochs = n_epochs, filename = "CNN_221108")

#%%
print('GRU network training start!!')
net_GRU = models['GRU'](input_dim, hidden_dim, layer_dim, output_dim).to("mps")
optimizer_GRU = optim.Adam(net_GRU.parameters())
training_networks(net = net_GRU, criterion = criterion, optimizer = optimizer_GRU, 
                  lr = lr, epochs = n_epochs, filename = "GRU_221108")
