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


def normalize_signal(signal):
    signal = signal - np.mean(signal)
    return signal / np.max(np.abs(signal))


def read_dataset(filename):
    return pd.read_csv(filename, header = None)



def read_csv_dataset(filename):
    dataset = []
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            print(row)
            input()

def write_dataset(dataset, filename):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in dataset:
            writer.writerow(row)
    return

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
    return res, inputs1, inputs2


def build_full_dataset(dataset):
    full_dataset = []
    i = 0
    while i < dataset.shape[0]-3:
        input1 = dataset.iloc[i]
        input2 = dataset.iloc[i+1]
        labels = dataset.iloc[i+2]
        labels, input1, input2 = build_labels(input1, input2, labels)
        full_dataset.append([input1, input2, labels])
        i += 3
    return full_dataset


def build_batches(dataset, batch_size):
    iterations = int(len(dataset) / batch_size)
    batches = []
    for i in range(iterations):
        bch = np.asarray(random.sample(dataset, batch_size))
        batches.append(np.asarray(random.sample(dataset, batch_size)))
    return np.asarray(batches)


def train_test_merger(dataset, proportion):
    print("DATASET LENGTH ", len(dataset))
    split_index = int(len(dataset) * proportion)
    train = dataset[:split_index]
    test = dataset[split_index]
    return train, test

def save_splitted(dataset):
    splitter = round(int(dataset.shape[0] * 0.8) / 3) * 3

    train = dataset.iloc[:splitter]
    test = dataset.iloc[splitter:]
    train.to_csv('train.csv', header=False, index=False)
    test.to_csv('test.csv', header=False, index=False)
    return


class RNN(nn.Module):
    def __init__(self, input_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
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
output_size = 2
batch_size = 5
n_layers = 2
seq_len = 1

rnn = RNN(input_size, output_size, n_layers=n_layers)
criterion = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.1, momentum=0.9)

dataset = read_dataset("train.csv")
# save_splitted(dataset)
print("DATASET LOADED")
data = np.asarray([])
labels = np.asarray([])


epoch = 1
running_loss = 0.0
hidden = None
i = 0

print("Building dataset")
final_dataset = build_full_dataset(dataset)

# train = read_dataset('train.csv')
# test = read_dataset('test.csv')
print("Building batches")
batches = build_batches(final_dataset, 3)
print(batches)

print("Training")
for epoch in range(2):
    i = 0
    losses = []
    steps = []
    for i, batch in enumerate(batches):
        input_batch = batch[:, :2]
        output_batch = batch[:, 2]
        inputs = torch.from_numpy(np.asarray(input_batch)).float()
        labels = torch.from_numpy(np.asarray(output_batch)).float()
        optimizer.zero_grad()
        res1 = rnn.forward(inputs)
        to_plot = normalize_signal(res1.detach().numpy()[0][0])
        to_plot2 = normalize_signal(input_batch[0])

        res = res1.squeeze()
        loss = criterion(res, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses.append(loss.item())
        steps.append(i)
        if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                plt.plot(range(len(to_plot2)), to_plot2)
                plt.plot(range(len(to_plot)), to_plot)
                plt.plot(range(len(to_plot2)), to_plot2)
                plt.plot(range(len(output_batch[0])), output_batch[0])
                plt.legend()
                plt.show()

    plt.plot(steps, losses)
    plt.show()

torch.save(rnn.state_dict(), 'model.pth')
