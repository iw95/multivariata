import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Kmeans:
    def __init__(self, data, k, iterations=5, labels=None):
        self.data = data
        self.datashape = data.shape
        self.k = k
        # initialize
        rng = np.random.default_rng(12345)
        self.labels = np.zeros((self.datashape[0]))
        self.means = rng.choice(data, k, replace=False)
        # clustering for <iterations> iterations
        self.cluster(iterations)
        # return labels and means
        return

    def cluster(self, iterations):
        dist = np.zeros((self.datashape[0],self.k))
        new_means = np.zeros((self.k,self.datashape[1]))
        for i in range(iterations):
            # recompute labels
            for k in range(self.k):
                dist[:,k] = np.linalg.norm(self.data - np.outer(np.ones((self.datashape[0],1)), self.means[k]), axis=1)
            self.labels = np.argmin(dist, axis=1)
            # visualize
            #self.visualize(i)
            # recompute means
            for k in range(self.k):
                la_k = self.labels==k
                new_means[k] = np.sum(self.data[la_k], axis=0) / np.sum(la_k)
            # compute change between means
            change = np.abs(new_means-self.means)
            self.means = new_means
        return

    def visualize(self, iteration):
        colours = mcolors.BASE_COLORS
        col_iter = iter(colours)
        dim1, dim2 = 0, 1
        fig = plt.figure()
        for k in range(self.k):
            c = next(col_iter)
            kdata = self.data[self.labels==k]
            plt.plot(kdata[:,dim1], kdata[:,dim2], c+'.')
            plt.plot(self.means[k,dim1], self.means[k,dim2], c+'o')
            # save figure
        plt.savefig(f'img/kmeans_iter{iteration}.png')
