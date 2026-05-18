# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:36:40 2021

A Novel Method for Virtual Real-Time Cumuliform Fluid Dynamics Simulation Using Deep Recurrent Neural Networks
Mathematics-MDPI paper at: https://doi.org/10.3390/math13172746 

    Author:
    * Sergio Calo Oliveira
"""

import torch
import torch.nn as nn


# We select CUDA so the code runs on the GPU if available
is_cuda = input('Use cuda? (Type yes or no)')

if is_cuda == 'yes':
    device = torch.device("cuda")
    print("GPU is available")
elif is_cuda == 'no':
    device = torch.device("cpu")
    print("GPU not available, CPU used")
else:
    print('Type yes or no')


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()

        # Network hyperparameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                            num_layers=n_layers, batch_first=True)
           
        # Linear classification layer (despite the name, in this case it performs regression rather than classification)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initialise the hidden state
        hidden, cell = self.init(batch_size)

        # Execution of the LSTM layer
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
       # print('out: ', out.shape)
        out = out[:,-1]
        #print('out: ', out.shape)
        
        # Execution of the linear layer
        out = self.fc(out)
        
        return out
    
    # Initialisation function
    def init(self, batch_size):

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        return hidden, cell

# Simple RNN (the structure is similar; the main difference is the recurrent layer, which in this case changes from LSTM to RNN)
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        
        out, hidden = self.rnn(x, hidden)

        out = out[:,-1]
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden
