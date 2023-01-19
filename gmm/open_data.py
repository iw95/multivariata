import numpy as np


def get_data():
    data = np.genfromtxt('data/selected_data.csv', delimiter=',')
    data = data[1:] # discard header
    data_lab = data[:,0]
    data = np.array(data[:,1:], dtype=float) # remove label and cast to float
    assert np.nan not in data
    return data
