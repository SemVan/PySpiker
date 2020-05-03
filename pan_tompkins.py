import numpy as np
from scipy.signal import convolve
from scipy.signal import medfilt
from matplotlib import pyplot as plt
from peakdetect import peakdetect
from scipy.interpolate import interp1d
from general_math import *
import pandas as pd
import csv

PERIOD = 0.04
FS = 1 / PERIOD

def pan_tompkins_algo(signal_to_process):
    sig_interp, newx = interpolation(signal_to_process, 10)
    sig_interp = butter_bandpass_filter(sig_interp, 0.1, 10, FS, 5)
    norm_signal = normalize(sig_interp)
    deriv = get_derivative(norm_signal)
    sq_deriv = get_square(deriv)
    integr = get_integration(sq_deriv, 80)
    integr = normalize(integr)
    norm_signal = normalize(norm_signal)
    return integr, norm_signal

def contacless_algo(csignal):
    csignal = medfilt(csignal, 3)
    csignal, newx = interpolation(csignal, 10)
    csignal = butter_bandpass_filter(csignal, 0.1, 10, FS, 5)
    norm_signal = normalize(csignal)
    return norm_signal

def get_derivative(signal):
    kernel = np.asarray([-1, 0, 1])
    out_signal = convolve(signal, kernel)
    return out_signal[:-80]


def get_square(signal):
    return np.square(signal)


def get_integration(signal, window_size):
    kernel = np.ones(window_size)
    out_signal = convolve(signal, kernel)
    return out_signal


def normalize(signal):
    s = signal / np.max(np.abs(signal))
    s = s - np.mean(s)
    return s


def reverse_signal(signal):
    signal = np.max(signal)-signal
    return signal


def interpolation(signal, ratio):
    f = interp1d(range(len(signal)), signal, kind='cubic')
    step = 1.0 / ratio
    newx = np.linspace(0, len(signal)-1, len(signal)*ratio)
    return f(newx), newx


def hard_peaks(signal):
    signal = np.asarray(signal) / np.max(signal)
    peaks2 = peakdetect(signal,lookahead=31)
    peaks_max = []
    for i in peaks2[0]:
        peaks_max.append(i[1])

    peaks_max_ind = []
    for j in peaks2[0]:
        peaks_max_ind.append(j[0])

    peaks_min = []
    for i in peaks2[1]:
        peaks_min.append(i[1])

    peaks_min_ind = []
    for j in peaks2[1]:
        peaks_min_ind.append(j[0])
    x1 = np.arange(0, len(signal))
    y1 = np.array(signal)
    return np.asarray(peaks_max_ind)


def from_new_x_to_old_x(peak_idx):
    to_old_idx = peak_idx / 10
    return to_old_idx.astype(np.int) - 1

def prepare_indices(sg, ind):
    full_ind = []
    for i in range(len(sg)):
        if i in ind:
            full_ind.append(1)
        else:
            full_ind.append(0)
    return full_ind


def read_dataset(filename):
    df = pd.read_csv(filename)
    return df

def write_dataset(dataset, filename):
    with open (filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in dataset:
            writer.writerow(row[0])
            writer.writerow(row[1])
            writer.writerow(row[2])
    return

def full_pan_tompkins(signal, cless_signal):
    res, signal_trans = pan_tompkins_algo(signal)
    cless_res, cless_signal = pan_tompkins_algo(cless_signal)

    indices = hard_peaks(res)
    maxes = res[indices]
    old_idx = from_new_x_to_old_x(indices)

    max_dots = signal_trans[indices]
    cless_dots = cless_signal[indices]
    full_indices = prepare_indices(cless_signal,indices)
    return signal_trans, res[:len(cless_signal)], full_indices


df = read_dataset("dataset.csv")
new_df = []



for i in range(df.shape[0] - 1):
    print("processing {} pair".format(i))
    signal, pt, true_indices = full_pan_tompkins(df.iloc[i+1], df.iloc[i])
    new_df.append([signal, pt, true_indices])
    i += 1

mixer = []
sz = 350
overlap = int(350/2)
for row in new_df:
    start = 0
    while start <= len(row[0]) - sz:
        r1 = row[0][start : start + sz]
        r2 = row[1][start : start + sz]
        r3 = row[2][start : start + sz]
        mixer.append([r1, r2, r3])
        start += overlap


print("total_inputs ", len(mixer))
write_dataset(mixer, "dataset_contact.csv")
