import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import L1Loss

def get_loss_batch3(obatch, lbatch):
    multi = obatch * lbatch
    bad = obatch - multi
    loss_step = (torch.sum(obatch - obatch * lbatch) / torch.sum(obatch))
    return loss_step / (obatch.shape[0] * 10)

def get_loss_batch2(obatch, lbatch):
    loss = 0
    for i in range(len(obatch)):
        answer = obatch[i]
        label = lbatch[i]
        multi = answer*label
        bad = answer - multi
        loss_step = (torch.sum(bad) / torch.sum(answer))
        loss += loss_step/(len(obatch) * 10)
        # print(loss_step)
        # plt.plot(range(len(bad.detach().numpy()[0])), bad.detach().numpy()[0], label="bad")
        # plt.plot(range(len(label.detach().numpy())), label.detach().numpy(), label="label")
        # plt.plot(range(len(multi.detach().numpy()[0])), multi.detach().numpy()[0], label="multi")
        # plt.plot(range(len(answer.detach().numpy()[0])), answer.detach().numpy()[0], label="answer")
        # plt.legend()
        # plt.show()
    return loss


def get_loss_batch(obatch, lbatch):
    loss = torch.zeros(len(obatch), requires_grad = True)
    for i in range(len(obatch)):
        loss[i] = get_loss(obatch[i], lbatch[i])
    return torch.sum(loss)

def get_loss(nnout, labels):
    # nnout = cutout_low(nnout, 0.6)
    # peaks = get_simple_peaks(nnout)
    peaks = torch.where(nnout == 1)[0]
    lpeaks = torch.where(labels == 1)[0]
    return loss_function(peaks, lpeaks)


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

def cutout_low(signal, threshold):
    idxs = torch.where(signal < threshold)
    signal[idxs] = 0
    return signal


def normalize_signal(signal):
    signal = signal - np.mean(signal)
    return signal / np.max(np.abs(signal))


def get_simple_peaks(signal):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i]>signal[i-1] and signal[i]>signal[i+1]:
            peaks.append(i)
    return peaks


def loss_function(pout, plab):
    first_part = 1-torch.exp(-torch.abs(torch.tensor(pout.shape, dtype=torch.double) - torch.tensor(plab.shape, dtype=torch.double)))
    # pairs = get_pairs(pout, plab[0])
    # diffs = get_pairs_diff(pairs)
    # second_part = np.sum(diffs)/710
    return first_part


def get_pairs(p1, p2):
    pairs = []

    for i in range(len(p1)):
        diff_min = 10000
        idx = 0
        for j in range(len(p2)):
            diff = np.abs(p1[i] - p2[j])
            if diff < diff_min:
                diff_min = diff
                idx = j
        pairs.append([i, idx])
    return pairs


def get_pairs_diff(ps):
    return [np.abs(x[0]-x[1]) for x in ps]

def rect(size):
    rect = np.zeros(size)
    rect[int(size/2)-5:int(size/2)+5] = 1
    return rect
