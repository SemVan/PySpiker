import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Sigmoid
import torch.optim as optim
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import gaussian
import random
import csv
import time

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=63, padding=31)
        self.conv2 = nn.Conv1d(hidden_size, 2*hidden_size, kernel_size=11, padding=5)
        # self.conv3 = nn.Conv1d(128, 1, kernel_size = 11, padding=5)
        
        self.lstm1 = nn.LSTM(2*hidden_size, 2*hidden_size, 
                            batch_first=True, 
                            bidirectional=True, 
                            num_layers=2, 
                            dropout=0.001)
        self.lstm2 = nn.LSTM(4*hidden_size, output_size, 
                            batch_first=True, 
                            bidirectional=True, 
                            num_layers=2, 
                            dropout=0.001)
        self.linear = nn.Linear(2*output_size, output_size, )


    def forward(self, inputs, hiddenLSTM1, hiddenLSTM2):
        batch_size = inputs.size(0)
        sigm = Sigmoid()
        
        
        c = self.conv1(inputs)
        p = F.relu(c)
        c = self.conv2(p)
        p = F.relu(c)
#         c = self.c3(p)

        # c = inputs
        
        p = c.transpose(1,2)
        c, hiddenLSTM1 = self.lstm1(p, hiddenLSTM1)
        p = torch.tanh(c)
        
        c, hiddenLSTM2 = self.lstm2(p, hiddenLSTM2)
        p = torch.tanh(c)
        
        c = self.linear(p)
        c = c.transpose(1,2)
        
        p = sigm(c)

        return p, hiddenLSTM1, hiddenLSTM2



input_size = 2
hidden_size = 32
output_size = 1
# batch_size = 3
n_layers = 2
seq_len = 1

hiddenLSTM1 = None
hiddenLSTM2 = None

rnn = RNN(input_size, hidden_size, output_size, n_layers=n_layers)
criterion = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.05, momentum=0.9)
