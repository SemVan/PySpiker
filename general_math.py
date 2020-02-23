from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.signal import correlate
from scipy.signal import butter, lfilter, lfilter_zi, welch, convolve2d
from scipy.stats import mannwhitneyu


def full_signals_procedure(ch1, ch2):
    ch1 = butter_bandpass_filter(ch1, 0.1, 5, 1000, 3)
    ch2 = butter_bandpass_filter(ch2, 0.1, 5, 1000, 3)
    ch1 = norm_signal(ch1)
    ch2 = norm_signal(ch2)
    # get_spectra(sig1)
    sh = get_phase_shift(ch1, ch2)
    print(sh)
    return ch1, ch2, sh


def get_channels_sum(frame_sequence):
    """Shape was frame-channel-row-column. Became row-column-frame"""

    f_seq_t = np.transpose(frame_sequence, (1, 0, 2, 3))
    weighted_sum = f_seq_t[0]/(f_seq_t[1]+f_seq_t[2])
    weighted_sum[weighted_sum == np.inf] = 0
    true_signal = np.transpose(weighted_sum, (1, 2, 0) )
    return true_signal

def plot_signals(ch1, ch2, offset):
    if offset>=0:
        ch2 = ch2[int(offset):]
    else:
        ch1 = ch1[:int(offset)]
    plt.plot(range(len(ch1)), ch1, color='red')
    plt.plot(range(len(ch2)), ch2, color='green')
    plt.show()
    return


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, z = lfilter(b, a, data, zi=zi*data[0])
    return y


def norm_signal(sp):
    max = np.max(sp)
    norm = []
    for i in range(len(sp)):
        norm.append(sp[i] / max)
    return norm


def get_mann_whitney_result(data):
    level = 23
    d_t = np.transpose(data)
    sh = d_t.shape
    u_test = np.zeros(shape = (sh[0], sh[0]))

    for i in range(sh[0]):
        for j in range(sh[0]):
            res, p = mannwhitneyu(d_t[i], d_t[j])
            u_test[i][j] = int(res<=level)
    print(u_test)

    return
