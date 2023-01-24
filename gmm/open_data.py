import numpy as np
import csv


def get_data(path='data/selected_data.csv'):
    data = np.genfromtxt(path, delimiter=',')
    data = data[1:] # discard header
    data_lab = data[:,0]
    data = np.array(data[:,1:], dtype=float) # remove label and cast to float
    assert np.nan not in data
    return data, get_header(path)


def get_header(path='data/selected_data.csv'):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            header = row[1:]
            break
    return header


def get_probs(k=3):
    data = np.genfromtxt(f'logs_{k}/pis.csv', delimiter=',')
    data = data[1:]  # discard header
    data = np.array(data[:, 1:], dtype=float)  # remove label and cast to float
    assert np.nan not in data
    return data # for each u: [p_u,i] mean_u argmax_u


def get_params(k=3, kmeans=False, first_setup=True):
    inittype = 'kmeans_init' if kmeans else 'random_init'
    setup = 'first_setup' if first_setup else 'second_setup'
    # data, header
    data, header = get_data(f'{setup}/data/selected_data.csv')
    # dim
    dims = len(header)
    # mu, sigma
    params = np.genfromtxt(f'{setup}/{inittype}/logs_{k}/params.csv', delimiter=',')
    params = params[-k:,:]
    mus = np.reshape(params[:,2+k:2+k+dims],(k,dims))
    sigmas = np.reshape(params[:,2+k+dims:],(k,dims,dims))
    # maxdims
    maxdims = np.zeros((dims, 2))
    for d in range(dims):
        maxdims[d] = np.array([np.min(data[:, d]), np.max(data[:, d])])
    # data labels dims mu sigma maxdims
    return data, header, dims, mus, sigmas, maxdims
