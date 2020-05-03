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
import argparse
from nnmath import *
from metrics import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str)
    parser.add_argument('--debug', dest='debug', type=str)
    return parser.parse_args()


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
    offset = 10
    kernel = gaussian(35, 5)
    inputs1 = normalize_signal(inputs1)
    inputs2 = normalize_signal(inputs2)
    res = np.convolve(labels, kernel, 'same')
    return res[offset:-offset], inputs1[offset:-offset], inputs2[offset:-offset]


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
    debug = dataset.iloc[:150]
    train.to_csv('train.csv', header=False, index=False)
    test.to_csv('test.csv', header=False, index=False)
    test.to_csv('debug.csv', header=False, index=False)

    return


class RNN(nn.Module):
    def __init__(self, input_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.c1 = nn.Conv1d(input_size, 16, kernel_size=63, padding=31, padding_mode='zeros')
        self.c2 = nn.Conv1d(16, 32, kernel_size=21, padding=10, padding_mode='zeros')
        self.c3 = nn.Conv1d(32, 32, kernel_size=21, padding=10, padding_mode='zeros')
        self.c4 = nn.Conv1d(32, 1, kernel_size=21, padding=10, padding_mode='zeros')
        self.c5 = nn.Conv1d(1, 1, kernel_size=21, padding=10, padding_mode='zeros')
        self.c6 = nn.Conv1d(1, 1, kernel_size = 21, padding=10, padding_mode='zeros')
        # self.l = nn.Linear(128, 1)


    def forward(self, inputs):
        batch_size = inputs.size(0)
        sigm = Sigmoid()
        c = self.c1(inputs)
        p = F.relu(c)
        c = self.c2(p)
        p = F.relu(c)
        c = self.c3(p)
        p = F.tanh(c)
        c = self.c4(p)
        p = F.tanh(c)
        c = self.c5(p)
        p = F.tanh(c)
        c = self.c6(p)
        p = F.tanh(c)
        return p



input_size = 2
output_size = 2
batch_size = 5
n_layers = 2
seq_len = 1

rnn = RNN(input_size, output_size, n_layers=n_layers)
criterion = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.9)

# optimizer = optim.Adadelta(rnn.parameters(), lr=1.0)

args = get_arguments()

# dataset = read_dataset("dataset_contact.csv")
# save_splitted(dataset)

if args.debug == "on":
    train_dataset = read_dataset("debug.csv")
    train_dataset = build_full_dataset(train_dataset)
    batches = build_batches(train_dataset, 3)
else:
    train_dataset = read_dataset("train.csv")
    test_dataset = read_dataset("test.csv")
    train_dataset = build_full_dataset(train_dataset)
    test_dataset = build_full_dataset(test_dataset)
    batches = build_batches(train_dataset, 9)
    t_batches = build_batches(test_dataset, 1)


print("DATASET LOADED")
data = np.asarray([])
labels = np.asarray([])

epoch = 1
running_loss = 0.0
hidden = None
i = 0

print("Building dataset")

print("Building batches")
running_custom_loss = 0

if args.mode == "train":
    print("Training")
    losses = []
    custom_losses = []
    steps = []
    j = 0
    for epoch in range(5):
        i = 0
        for i, batch in enumerate(batches):
            input_batch = batch[:, :2]
            output_batch = batch[:, 2]
            inputs = torch.from_numpy(np.asarray(input_batch)).float()
            labels = torch.from_numpy(np.asarray(output_batch)).float()
            optimizer.zero_grad()
            res1 = rnn.forward(inputs)
            res = res1.squeeze()

            custom_loss = get_loss_batch3(res1, labels)

            loss = criterion(res, labels)
            print(custom_loss)
            custom_loss.backward()
            print(res.grad)
            optimizer.step()
            running_loss += loss.item()
            running_custom_loss += custom_loss
            losses.append(loss.item())
            custom_losses.append(custom_loss.item())
            steps.append(j)
            j += 1
            if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                    print(running_custom_loss/100)
                    running_custom_loss = 0.0

    plt.plot(steps, losses, label="MSE")
    plt.plot(steps, custom_losses, label="CUSTOM LOSS")
    plt.show()
    torch.save(rnn.state_dict(), 'model.pth')
else:
    print("Testing")
    counts = []
    rnn.load_state_dict(torch.load('model.pth'))
    for i, batch in enumerate(t_batches):
        input_batch = batch[:, :2]
        output_batch = batch[:, 2]
        inputs = torch.from_numpy(np.asarray(input_batch)).float()
        labels = torch.from_numpy(np.asarray(output_batch)).float()
        res1 = rnn.forward(inputs)

        custom_loss = get_loss_batch2(res1, labels)

        to_plot = res1.detach().numpy()[0][0]
        to_plot2 = normalize_signal(input_batch[0])
        r = get_hr_metric(to_plot, output_batch[0])
        if len(r) >0:
            counts.append(get_hr_metric(to_plot, output_batch[0]))
        # plt.plot(range(len(to_plot)), to_plot)
        # plt.plot(range(len(output_batch[0])), output_batch[0])
        # plt.legend()
        # plt.show()

        res = res1.squeeze()
        loss = criterion(res, labels)
        print(loss)
    counts = np.asarray(counts)
    print("AVERAGE HR DIFF ", np.nanmean(counts[:, 3]))
    print(counts[:, 3])
    print("PERCENTAGE OF GOOD PIECES ", np.mean(counts[:, 2]))
