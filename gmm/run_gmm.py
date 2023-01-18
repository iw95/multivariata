import numpy as np
from scipy.stats import multivariate_normal
import GMM
import kmeans
import matplotlib.pyplot as plt
import os


def normal_data():
    # generating multivariate data
    mnd1 = multivariate_normal(np.array([0, 0]), np.array([[2, 0], [0, 3]]))
    mnd2 = multivariate_normal(np.array([5, 2]), np.array([[1, 0.75], [0.75, 2]]))
    n1 = 50
    n2 = 70
    data = np.concatenate((mnd1.rvs(n1), mnd2.rvs(n2)), axis=0)
    dist = lambda x: (n1 / (n1 + n2)) * mnd1.pdf(x) + (n2 / (n1 + n2)) * mnd2.pdf(x)
    visualize(data, 2, dist)
    return data


def visualize(data, k, dist):
    # choose dimension to create 2-dimensional image
    dim1, dim2 = 0, 1
    # compute frame
    maxdims = np.zeros((data.shape[1], 2))
    for d in range(data.shape[1]):
        maxdims[d] = np.array([np.min(data[:, d]), np.max(data[:, d])])
    # create meshgrid
    gran = 100
    mg1 = np.linspace(maxdims[dim1, 0], maxdims[dim1, 1], gran)
    mg2 = np.linspace(maxdims[dim2, 0], maxdims[dim2, 1], gran)
    x, y = np.meshgrid(mg1, mg2)
    pos = np.dstack((x, y))
    # create plot
    fig = plt.figure()
    # calculate probability distribution
    z = dist(pos)
    # plot heat map and data points
    plt.contourf(x, y, z)
    plt.plot(data[:, dim1], data[:, dim2], 'k.')
    # save figure
    plt.savefig(f'img/gmm_orig_dist.png')


def gmm():
    data = normal_data()
    clustering = GMM.GMM(data, 2, threshold=0.01)


def kmeaning():
    data = normal_data()
    km = kmeans.Kmeans(data, 2, iterations=3)


def gmm_on_kmeans():
    k = 2
    data = normal_data()
    km = kmeans.Kmeans(data, k, iterations=3)
    probs = np.concatenate([np.expand_dims(np.array(km.labels == i, dtype=float), axis=1) for i in range(k)], axis=1)
    gmm = GMM.GMM(data, k=k, threshold=0.01, probs=probs)


def clear_img():
    # ATTENTION: removes all png files in subfolder img/
    os.system('rm img/*.png')


def main():
    clear_img()
    #gmm_on_kmeans()
    gmm()


if __name__ == "__main__":
    main()
