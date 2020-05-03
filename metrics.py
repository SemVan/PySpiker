import numpy as np
from matplotlib import pyplot as plt

def get_hr_metric(output, label):
    output = cutout_low(np.asarray(output), 0.7)
    out_idx = get_simple_peaks(output)

    label = np.asarray(label)
    lbl_idx = np.where(label==1)[0]
    lbl_hrs = []
    for i in range(len(lbl_idx)-1):
        lbl_hrs.append(lbl_idx[i+1] - lbl_idx[i])
    lbl_hr = np.mean(lbl_hrs) * 0.04/10 * 60

    out_hrs = []
    for i in range(len(out_idx)-1):
        out_hrs.append(out_idx[i+1] - out_idx[i])
    out_hr = np.mean(out_hrs) * 0.04/10 * 60
    good = int(abs(lbl_hr - out_hr) <= 1)
    if np.isnan(out_hr) or np.isnan(lbl_hr):
        return []
    return [lbl_hr, out_hr, good, abs(lbl_hr-out_hr)]


def cutout_low(signal, threshold):
    idxs = np.where(signal < threshold)
    signal[idxs] = 0
    return signal

def get_simple_peaks(signal):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i]>signal[i-1] and signal[i]>signal[i+1]:
            peaks.append(i)
    return peaks
