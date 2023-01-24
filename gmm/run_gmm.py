import numpy as np
from scipy.stats import multivariate_normal
import GMM
import kmeans
from open_data import get_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def gen_normal_data():
    # generating multivariate data
    mnd1 = multivariate_normal(np.array([0, 0]), np.array([[2, 0], [0, 3]]))
    mnd2 = multivariate_normal(np.array([3, 2]), np.array([[1, 0.75], [0.75, 2]]))
    n1 = 50
    n2 = 70
    data = np.concatenate((mnd1.rvs(n1), mnd2.rvs(n2)), axis=0)
    dist = lambda x: (n1 / (n1 + n2)) * mnd1.pdf(x) + (n2 / (n1 + n2)) * mnd2.pdf(x)
    visualize(data, 2, dist=dist, dist1=mnd1.pdf, dist2=mnd2.pdf)
    return data


def gen_normal_data1():
    # generating multivariate data
    mnd = multivariate_normal(np.array([3, 2]), np.array([[1, -0.75], [-0.75, 2]]))
    n = 120
    dist = mnd.pdf
    data = mnd.rvs(n)
    visualize(data, 1, dist=dist, dist1=dist, dist2=dist)
    return data


def visualize(data, k, dist, dist1, dist2): # Filled contour plot
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
    plt.savefig(f'img/gmm_filled_orig.png')
    plt.close()

    colours = mcolors.BASE_COLORS
    col_iter = iter(colours)
    fig = plt.figure()
    # distribution 1
    z1 = dist2(pos)
    plt.contour(x, y, z1, colors=next(col_iter))
    # distribution 2
    z2 = dist1(pos)
    plt.contour(x, y, z2, colors=next(col_iter))
    plt.plot(data[:, dim1], data[:, dim2], 'k.')
    # save figure
    plt.savefig(f'img/gmm_cont_orig.png')
    plt.close()


def gmm(data=None, k=None, threshold=None, labels=None):
    if data is None:
        data = gen_normal_data()
    if k is None:
        k = 2
    if threshold is None:
        threshold = 0.01
    clustering = GMM.GMM(data=data,k=k, threshold=threshold, labels=labels)


def kmeaning(data=None, k=None, iterations=None, labels=None):
    if data is None:
        data = gen_normal_data()
    if k is None:
        k = 2
    if iterations is None:
        iterations = 3
    km = kmeans.Kmeans(data=data, k=k, iterations=iterations, labels=labels)


def gmm_on_kmeans(data=None, k=None, iterations=None, threshold=None, labels=None):
    if data is None:
        data = gen_normal_data()
    if k is None:
        k = 2
    if iterations is None:
        iterations = 3
    if threshold is None:
        threshold = 0.01
    km = kmeans.Kmeans(data=data, k=k, iterations=iterations, labels=labels)
    print(f'kmeans done for k={k}')
    probs = np.concatenate([np.expand_dims(np.array(km.labels == i, dtype=float), axis=1) for i in range(k)], axis=1)
    gmm = GMM.GMM(data=data, k=k, threshold=threshold, probs=probs, labels=labels)


def clear_img():
    # ATTENTION: removes all png files in subfolder img/
    os.system('rm img/*.png')


def main():
    #clear_img()
    #gmm_on_kmeans()
    data, labels = get_data()
    for k in range(2,8):
        print(f'starting with k={k}')
        gmm(data=data, k=k, threshold=0.5, labels=labels)
        print('done')


if __name__ == "__main__":
    main()
