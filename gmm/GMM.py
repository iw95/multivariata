import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class GMM:

    def __init__(self, data, k, threshold=0.1, probs=None):
        # initializing basic variables
        self.threshold = threshold
        self.k = k
        self.data = data
        shape = data.shape
        self.n = shape[0]
        self.dim = shape[1]
        self.maxdims = np.zeros((self.dim,2))
        for d in range(self.dim):
            self.maxdims[d] = np.array([np.min(data[:,d]),np.max(data[:,d])])
        # initializing parameters
        self.det = np.zeros(k)
        if probs is None:
            self.post_probs = np.zeros((self.n,k))
            self.cl_probs = np.zeros(k)
            self.mu, self.sigma = self.init_params()
        else:
            assert(probs.shape == (self.n,k))
            self.post_probs = probs
            self.mu = np.zeros((k,self.dim))
            self.sigma = np.zeros((k,self.dim,self.dim))
            self.maximize()

        self.clustering()

    def init_params(self):
        return self.rand_init()

    def rand_init(self):
        self.cl_probs = np.ones((self.k)) * (1 / self.k)
        rng = np.random.default_rng(12345)
        # creating positive definite sigmas (!)
        sigmas = [np.expand_dims(np.diag(rng.random(self.dim)), axis=0) for k in range(self.k)]
        return rng.choice(self.data,self.k,replace=False), np.concatenate(sigmas)

    def update_det(self):
        for i in range(self.k):
            self.det[i] = np.linalg.det(self.sigma[i,:,:])
        return

    def density(self,k,x): # not using a library
        self.update_det()
        factor = 1/(np.sqrt((2*np.pi)**k * self.det[k]))
        return factor * np.exp(-0.5 * (np.transpose(x-self.mu[k]) @ np.linalg.inv(self.sigma[k]) @ (x-self.mu[k])))

    def density_scipy(self,k,x): # using scipy
        mnd = multivariate_normal(self.mu[k], self.sigma[k])
        return mnd.pdf(x) # probability density function

    def estimate(self):
        # Computes posterior probabilities for unit u to belong to cluster i for all u,i
        old_post_prob = np.copy(self.post_probs)
        # update posterior probabilities
        for u in range(self.n):
            comp_dens = np.array([self.density_scipy(i,self.data[u]) for i in range(self.k)]) # shape (k,)
            divisor = np.dot(comp_dens, self.cl_probs)
            self.post_probs[u] = comp_dens*self.cl_probs / divisor
        return abs(old_post_prob - self.post_probs) # return changes to evaluate convergence

    def maximize(self):
        # compute new probabilities for each cluster
        sum_per_cluster = np.sum(self.post_probs,axis=0)
        self.cl_probs = sum_per_cluster / self.n
        # compute new means and variances
        for k in range(self.k):
            # new means
            self.mu[k] = np.sum(np.diag(self.post_probs[:,k]) @ self.data,axis=0) / sum_per_cluster[k]
            # new variances
            self.sigma[k] = np.sum([self.post_probs[u,k] * np.outer(self.data[u]-self.mu[k], self.data[u]-self.mu[k]) for u in range(self.n)],axis=0)
            self.sigma[k] = self.sigma[k] / sum_per_cluster[k]
        return

    def step(self):
        changes = self.estimate()
        if np.max(changes) < self.threshold:
            # convergence reached
            return False # Continue = False
        self.maximize()
        return True # Continue = True

    def visualize(self,iteration):
        # choose dimension to create 2-dimensional image
        dim1, dim2 = 0, 1
        # create meshgrid
        gran = 100
        mg1 = np.linspace(self.maxdims[dim1,0],self.maxdims[dim1,1], gran)
        mg2 = np.linspace(self.maxdims[dim2,0],self.maxdims[dim2,1], gran)
        x, y = np.meshgrid(mg1, mg2)
        pos = np.dstack((x, y))
        # create plot
        fig = plt.figure()
        # calculate probability distribution
        z = np.zeros((gran,gran))
        for k in range(self.k):
            print(f'Sigma for cluster {k} in iteration {iteration}\n{self.sigma[k]}')
            rv = multivariate_normal(self.mu[k], self.sigma[k])
            z += self.cl_probs[k] * rv.pdf(pos)
        # plot heat map and data points
        plt.contourf(x, y, z)
        plt.plot(self.data[:,dim1], self.data[:,dim2],'k.')
        # save figure
        plt.savefig(f'img/gmm_iter{iteration}.png')
        plt.close()

        # plot contour lines
        # reuse x, y, pos, mg1, mg2
        fig = plt.figure()
        colours = mcolors.BASE_COLORS
        col_iter = iter(colours)
        for k in range(self.k):
            rv = multivariate_normal(self.mu[k], self.sigma[k])
            z = rv.pdf(pos)
            plt.contour(x,y,z,colors=next(col_iter))
        plt.plot(self.data[:,dim1], self.data[:,dim2],'k.')
        # save figure
        plt.savefig(f'img/gmm_cont_iter{iteration}.png')
        plt.close()

    def clustering(self):
        # Parameters already initialized in init
        iteration = 0
        self.visualize(iteration)
        while self.step():
            iteration = iteration + 1
            self.visualize(iteration)
        return iteration

