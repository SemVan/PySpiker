import csv
import numpy as np

DATASET_NAME = 'dataset.csv'

def read_dataset(file_name):
    
    signals = []
    contact = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
        i = -1
        for signal in reader:
            contactless = contact
            contact = signal
            i += 1
            if (i % 2) > 0:
                signals.append((np.asarray(contactless), 
                                np.asarray(contact)))   

    return signals


def GRU_get_training_set(signals, sample_length):
    
    training_set = []
    dataset_length = len(signals)
    signal_length = len(signals[0][0])
    n = signal_length % sample_length
    for (contactless,contact) in signals:
        for i in range(n):
            training_set.append( (contactless[i*sample_length : (i+1)*sample_length], 
                                  contact[i*sample_length : (i+1)*sample_length]) )
            
    return training_set