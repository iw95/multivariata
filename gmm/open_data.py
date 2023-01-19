import numpy as np
import csv


def get_data():
    data = np.genfromtxt('data/selected_data.csv', delimiter=',')
    data = data[1:] # discard header
    data_lab = data[:,0]
    data = np.array(data[:,1:], dtype=float) # remove label and cast to float
    assert np.nan not in data
    return data, get_header()


def get_header():
    with open('data/selected_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            header = row[1:]
            break
    return header