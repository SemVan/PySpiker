import csv
import numpy as np

DATASET_NAME = 'dataset.csv'

def read_dataset(file_name):
    contacts = []
    contactlesses = []
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quoting = csv.QUOTE_NONNUMERIC)
        i = 0
        for row in reader:
            i += 1
            if (i % 2) > 0:
                contactlesses.append(row)
            else:
                contacts.append(row)    

    return np.asarray(contacts), np.asarray(contactlesses)