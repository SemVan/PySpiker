import csv
import numpy as np
import random
import torch

DATASET_NAME = 'dataset.csv'

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


def GRU_get_training_set(signals, sample_length):
    
    training_set = []
    testing_set = []
    dataset_length = len(signals)
    training_length = 0.8*dataset_length
    signal_length = len(signals[0][0])
    n = signal_length // sample_length
    j = -1
    for (contactless,contact) in signals:
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