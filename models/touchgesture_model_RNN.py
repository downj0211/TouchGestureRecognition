import torch.nn as nn
import torch.nn.functional as F
import torch

class touchgesture_model_RNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_size = input_dim, hidden_size = hidden_dim, num_layers = layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        x_new = torch.zeros(x.size()[0], 10, 46).to("mps")
        for i in range(10):
            x_new[:,i,:] = x[:,10*i,:]
            
        x = x_new
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
#        print(out.size())
        out = self.fc(out[:, -1, :])
#        print(out)
        
#        # original code
#        x_new = torch.zeros(x.size()[0], 10, x.size()[2]).to("cuda")
#        for i in range(10):
#            x_new[:,i,:] = x[:,10*i,:]
#         
#        x = x_new
#        
#        h0 = self.init_hidden(x)
#        out, hn = self.rnn(x, h0)
##        print(out.size())
#        out = self.fc(out[:, -1, :])
        return out
    
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to("mps")
#        return [t.cuda() for t in (h0)]
#        return [t for t in (h0, c0)]
        return h0
    