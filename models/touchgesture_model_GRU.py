import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

class touchgesture_model_GRU(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
#        self.gru = nn.GRU(46, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
            
            
    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        
        # original code
#        h0 = self.init_hidden(x)
#        out, hn = self.gru(x, h0)
#        out = self.fc(out[:, -1, :])
        return out
#         return torch.sigmoid(out)
    
    def init_hidden(self, x):
        # zero 
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to("mps")
        
        # randn
#         h0 = torch.randn(self.layer_dim, x.size(0), self.hidden_dim).cuda()
    
        # xavier init
#         h0 = torch.empty(self.layer_dim, x.size(0), self.hidden_dim).cuda()
#         nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))


        # He init
#         h0 = torch.empty(self.layer_dim, x.size(0), self.hidden_dim).cuda()
#         nn.init.kaiming_uniform_(h0, mode='fan_in', nonlinearity='relu')

        return h0
    