import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class RNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, num_outputs):
        super(RNN, self).__init__()
        self.num_hiddens = num_hiddens #Number of neurons in the hidden layer
        self.num_layers = num_layers #Number of hidden layers
        self.rnn = nn.RNN(num_inputs, num_hiddens, num_layers, batch_first = True) #RNN layer includes inputs and hidden states
        self.fc = nn.Linear(num_hiddens, num_outputs) #Output
        
    def forward(self, X):
        #Initialising hidden state
        h0 = torch.zeros(self.num_layers, X.size(0), self.num_hiddens).requires_grad_()
        out, h_state = self.rnn(X, h0.detach())
        out = self.fc(out[:, -1, :]) 
        return out