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


def draw_result(res, inp, out):
    to_plot = normalize_signal(res.detach().numpy()[0][0])
    to_plot2 = normalize_signal(inp[0][0])
    to_plot3 = out[0]
    
    plt.plot(range(len(to_plot)), to_plot)
    plt.plot(range(len(to_plot2)), to_plot2)
    plt.plot(range(len(to_plot3)), to_plot3)
    plt.show()


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
    
def pearsonr(result, target):
    s = 0
    for i in range(len(result)):
        res = result[i] - torch.mean(result[i])
        tag = target[i] - torch.mean(target[i])
        if ((torch.sum(res**2)!=0) & (torch.sum(tag**2)!=0)):
            s += -1 + torch.sum(res*tag) / (torch.sum(res**2)*torch.sum(tag**2)) ** 0.5
        else:
            s += 1  
        
    return -s / (2*len(result))



def normalize_signal(signal):
    signal = signal - np.mean(signal)
    if np.max(np.abs(signal))>0:
        return signal / np.max(np.abs(signal))
    else:
        return signal


def read_dataset(filename):
    return pd.read_csv(filename, header = None)



def read_csv_dataset(filename):
    dataset = []
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # print(row)
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


dataset = read_dataset("train.csv")
# save_splitted(dataset)
print("DATASET LOADED")
data = np.asarray([])
labels = np.asarray([])

print("Building dataset")
final_dataset = build_full_dataset(dataset)

# train = read_dataset('train.csv')
# test = read_dataset('test.csv')
print("Building batches")
batches = build_batches(final_dataset, 3)
# print(batches)