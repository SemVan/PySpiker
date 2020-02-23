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


def normalize_signal(signal):
    signal = signal - np.mean(signal)
    return signal / np.max(np.abs(signal))

def read_dataset(filename):
    return pd.read_csv(filename, header = None)


def check_labels(labs):
    pos = -1
    for i, elem in enumerate(labs):
        if elem == 1:
            pos = i
    if abs(pos - len(labs)/2) <25:
        return 1
    else:
        return 0


def build_labels(inputs1, inputs2, labels):
    kernel = gaussian(21, 11)
    inputs1 = normalize_signal(inputs1)
    inputs2 = normalize_signal(inputs2)
    res = np.convolve(labels, kernel, 'same')
    return res[10:-10], inputs1[10:-10], inputs2[10:-10]



def build_batch(inputs1, inputs2, labels):
    n = 100
    step = 5
    data = []
    lab = []
    i = 100
    while i < len(inputs1)-n:
        data.append([np.asarray(inputs1[i:i+n]).astype(np.double), np.asarray(inputs2[i:i+n]).astype(np.double)])
        # data.append(np.asarray(inputs1[i:i+n]).astype(np.double))
        l = check_labels(labels[i:i+n])
        lab.append(l)
        # if l == 0:
        #     plt.plot(range(n), inputs1[i:i+n])
        #     plt.plot(range(n), inputs2[i:i+n])
        #     plt.show()
        i += step
    ones = np.sum(lab)
    final_data = []
    final_lab = []
    for i, elem in enumerate(data):
        # print(lab[i])
        if lab[i] == 0 and ones > 0:
            final_data.append(elem)
            final_lab.append(0)
            ones -= 1
        if lab[i] == 1:
            final_data.append(elem)
            final_lab.append(1)
    return final_data, final_lab


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.c1 = nn.Conv1d(input_size, 64, kernel_size=63, padding=31, padding_mode='zeros')
        self.c2 = nn.Conv1d(64, 128, kernel_size=11, padding=5, padding_mode='zeros')
        self.c3 = nn.Conv1d(128, 1, kernel_size = 11, padding=5, padding_mode='zeros')


    def forward(self, inputs):
        batch_size = inputs.size(0)

        c = self.c1(inputs)
        p = F.relu(c)
        c = self.c2(p)
        p = F.relu(c)
        c = self.c3(p)
        sigm = Sigmoid()
        p = sigm(c)
        return p



input_size = 2
hidden_size = 5
output_size = 2
batch_size = 5
n_layers = 2
seq_len = 1

rnn = RNN(input_size, hidden_size, output_size, n_layers=n_layers)
criterion = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.1, momentum=0.9)

dataset = read_dataset("dataset_contact.csv")
print("DATASET LOADED")
data = np.asarray([])
labels = np.asarray([])


epoch = 1
running_loss = 0.0
hidden = None
i = 0

for epoch in range(2):
    i = 0
    losses = []
    steps = []
    while i < dataset.shape[0]-3:

        input1 = dataset.iloc[i]
        input2 = dataset.iloc[i+1]
        labels = dataset.iloc[i+2]
        labels1, input1, input2 = build_labels(input1, input2, labels)

        input_batch = [[input1, input2]]
        output_batch = [labels1]
        inputs = torch.from_numpy(np.asarray(input_batch)).float()
        labels = torch.from_numpy(np.asarray(output_batch)).float()

        res1 = rnn.forward(inputs)
        to_plot = normalize_signal(res1.detach().numpy()[0][0])
        to_plot2 = normalize_signal(input1)
        # print(to_plot)
        # plt.plot(range(len(to_plot2)), to_plot2)
        # plt.plot(range(len(to_plot)), to_plot)
        # plt.plot(range(len(labels1)), labels1)
        # plt.legend()
        # plt.show()

        res = res1.squeeze()
        optimizer.zero_grad()

        loss = criterion(res, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        losses.append(loss.item())
        steps.append(i)
        if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                # plt.plot(range(len(to_plot2)), to_plot2)
                plt.plot(range(len(to_plot)), to_plot)
                plt.plot(range(len(to_plot2)), to_plot2)
                plt.plot(range(len(labels1)), labels1)
                plt.legend()
                plt.show()

        i += 3
    plt.plot(steps, losses)
    plt.show()
