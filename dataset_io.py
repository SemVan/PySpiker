import csv
import numpy as np
import random
import torch

DATASET_NAME = 'dataset.csv'
CONTACT_DATASET_NAME = 'dataset_contact.csv'

def read_dataset(file_name):
    
    signals = []
    contactless = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
        i = -1
        for signal in reader:
            contact = contactless
            contactless = signal
            i += 1
            if (i % 2) > 0:
                signals.append((np.asarray(contactless), 
                                np.asarray(contact)))   

    return signals


def read_contact_dataset(file_name):
    
    contact = []
    signals = []
    contactless = []
    peaks = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
        i = 0
        for signal in reader:
            contactless = contact
            contact = peaks
            peaks = signal
            i += 1
            if (i % 3) < 1:
                for j in range(7,len(peaks)-7):
                    if peaks[j]>0.9:
                        peaks[j-1] = 0.85
                        peaks[j-2] = 0.8
                        peaks[j-3] = 0.7
                        peaks[j-4] = 0.6
                        peaks[j-5] = 0.5
                        peaks[j-6] = 0.3
                        peaks[j-7] = 0.1
                        peaks[j+1] = 0.85
                        peaks[j+2] = 0.8
                        peaks[j+3] = 0.7
                        peaks[j+4] = 0.6
                        peaks[j+5] = 0.5
                        peaks[j+6] = 0.3
                        peaks[j+7] = 0.1
                signals.append((np.asarray(contactless), 
                                np.asarray(contact),
                                np.asarray(peaks)))   

    return signals


def GRU_get_training_set(signals, sample_length):
    
    training_set = []
    testing_set = []
    dataset_length = len(signals)
    training_length = 0.8*dataset_length
    signal_length = len(signals[0][0])
    n = signal_length // sample_length
    j = -1
    if len(signals[0])<3:
        for (contactless,contact) in signals:
            contactless /= np.max(contactless)
            contact /= np.max(contact)
            j += 1
            if j < training_length:
                for i in range(n):
                    training_set.append( (contactless[i*sample_length : (i+1)*sample_length], 
                                          contact[i*sample_length : (i+1)*sample_length]) )
            else:
                for i in range(n):
                    testing_set.append( (contactless[i*sample_length : (i+1)*sample_length], 
                                         contact[i*sample_length : (i+1)*sample_length]) )
    else:
        for (contactless, contact, peaks) in signals:
            # sample_length = len(contactless)
            contactless /= np.max(contactless)
            contact /= np.max(contact)
            # contactless = contact
            contact = peaks
            j += 1
            if j < training_length:
                for i in range(n):
                    training_set.append( (contactless[i*sample_length : (i+1)*sample_length], 
                                          contact[i*sample_length : (i+1)*sample_length]) )
            else:
                for i in range(n):
                    testing_set.append( (contactless[i*sample_length : (i+1)*sample_length], 
                                         contact[i*sample_length : (i+1)*sample_length]) )
                
    random.shuffle(training_set)
                
    return training_set, testing_set


def parse_set(t_set, batch_size):
    inputs = []
    outputs = []
    batch_in = []
    batch_out = []
    i = 0
    for (contactless,contact) in t_set:
        i += 1
        batch_in.append(contactless)
        batch_out.append(contact)
        if i % batch_size == 0:
            inputs.append(batch_in)
            outputs.append(batch_out)
            batch_in = []
            batch_out = []
        
    return torch.Tensor(inputs), torch.Tensor(outputs)


def parse_set2(t_set, batch_size):
    t_set2 = []
    batch_in = []
    batch_out = []
    i = 0
    for (contactless,contact) in t_set:
        i += 1
        batch_in.append(contactless)
        batch_out.append(contact)
        if i % batch_size == 0:
            t_set2.append((batch_in, batch_out))
            batch_in = []
            batch_out = []
        
    return t_set2


Signals = read_dataset(DATASET_NAME)
Training_set, Testing_set = GRU_get_training_set(Signals, 150)
outputs, inputs = parse_set(Training_set, 50)