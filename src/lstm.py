#LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim #Number of neurons in hidden layer
        self.num_layers = num_layers #Number of hidden layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) #LSTM layer
        self.fc = nn.Linear(hidden_dim, output_dim) #Output layer

    def forward(self, x):
        #Intialising hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        #Intialising cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach())) 
        out = self.fc(out[:, -1, :]) 
        return out