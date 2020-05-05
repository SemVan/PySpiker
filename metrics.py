import numpy as np
from matplotlib import pyplot as plt

def get_hr_metric(output, label, thresh):
    output = output.detach().numpy()[0][0]
    output = cutout_low(np.asarray(output), thresh)
    out_idx = get_simple_peaks(output)

    label = np.asarray(label)
    lbl_idx = np.where(label==1)[0]
    lbl_hr, lbl_diffs = get_hr(lbl_idx)
    out_hr, out_diffs = get_hr(out_idx)
    good = int(abs(lbl_hr - out_hr) <= 1)
    tf = get_true_false(lbl_idx, out_idx, len(label))
    print(tf)
    if np.isnan(out_hr) or np.isnan(lbl_hr):
        return [0, out_hr, 0, out_hr, tf]
    # plt.plot(range(len(output)), output)
    # plt.plot(range(len(label)), label)
    # plt.show()
    # tf = get_true_false(lbl_idx, out_idx, len(label))
    return [lbl_hr, out_hr, good, abs(lbl_hr-out_hr), tf]


def get_hr(indices):
    hrs = []
    for i in range(len(indices)-1):
        hrs.append(indices[i+1] - indices[i])
    hr = 60 / (np.mean(hrs) * 0.04)
    return hr, hrs


def cutout_low(signal, threshold):
    idxs = np.where(signal < threshold)
    signal[idxs] = 0
    m = np.max(signal)
    if m > 0:
        signal = signal / m
    return signal


def get_simple_peaks(signal):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i]>signal[i-1] and signal[i]>signal[i+1]:
            peaks.append(i)
    return filter_peaks(peaks)


def filter_peaks(peaks):
    new_peaks = []
    lst = False
    i = 0
    if len(peaks):
        while i < len(peaks)-1:
            if abs(peaks[i] - peaks[i+1]) < 10:
                new_peaks.append((peaks[i] + peaks[i+1])/2)
                i += 2
                lst = True
            else:
                new_peaks.append(peaks[i])
                lst = False
                i+= 1
        if not lst:
            new_peaks.append(peaks[-1])
    return new_peaks


def get_quadratic_approximation(signal):
    peaks = []
    for i in range(2, len(signal)-5, 5):
        dots = signal[i-2:i+3]
        x_dots = list(range(i-2, i+3))
        x = get_piece_approx(x_dots, dots)
        if x is not None:
            peaks.append(x)
    return peaks


def get_piece_approx(x, y):
    coeffs = np.polyfit(x, y, 2)
    extremum = -coeffs[1] / (2 * coeffs[0])
    second_deriv =  2 * coeffs[0]

    if extremum <= np.max(x) and extremum >= np.min(x) and second_deriv < 0:
        return extremum
    else:
        return None


def get_true_false(peaks_true, peaks_new, length):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    peaks_true_full = build_peaks_interval(peaks_true, length)
    peaks_new_full = build_peaks_interval(peaks_new, length)
    for i in range(length-1):
        if i in peaks_new:
            if peaks_true_full[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if i in peaks_true:
                if peaks_new_full[i] != 1:
                    fn += 1
                else:
                    tn += 1
            else:
                tn += 1

    return np.asarray([tp, tn, fp, fn])

def get_precision_recall(tp, tn, fp, fn):
    precision = float(float(tp) / (tp + fp))
    recall = float(tp) / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return [precision, recall, f1]

def build_peaks_interval(peaks, l):
    new_peaks = np.zeros(l)
    if len(peaks) > 0:
        min_peak = np.min(peaks)
        max_peak = np.max(peaks)
        for peak in peaks:
            new_peaks[max(0, int(peak) - 5): min(l-1, int(peak) + 5)] = 1
    return new_peaks
