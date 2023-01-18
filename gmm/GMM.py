import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class GMM:

    def __init__(self, data, k, threshold=0.1):
        self.threshold = threshold
        self.k = k
        self.data = data
        shape = data.shape
        self.n = shape[0]
        self.dim = shape[1]
        self.maxdims = np.zeros((k,2))
        for d in range(k):
            self.maxdims[d] = np.array([np.min(data[:,d]),np.max(data[:,d])])

        self.post_probs = np.zeros((self.n,k))
        self.cl_probs = np.zeros(k)
        self.mu, self.sigma = self.init_params()
        self.det = np.zeros(k)

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
            comp_dens = np.array([self.density_scipy(i,self.data[u]) for i in range(self.k)])
            divisor = np.max(np.expand_dims(comp_dens,axis=0) @ np.expand_dims(self.cl_probs,axis=1))
            for i in range(self.k):
                self.post_probs[u,i] = comp_dens[i]*self.cl_probs[i] / divisor

        #return False in old_post_prob == self.post_probs # True if at least one entry was changed
        return abs(old_post_prob - self.post_probs) # return changes to evaluate convergence

    def maximize(self):
        print('BEFORE MAXIMIZE')
        print(f'cl_probs: {self.cl_probs[0]}---\nmu: -----\n{self.mu[0]}\nsigma: ----\n{self.sigma[0]}')
        # compute new probabilities for each cluster
        sum_per_cluster = np.sum(self.post_probs,axis=0)
        self.cl_probs = sum_per_cluster / self.n
        print(f'cl_probs: {self.cl_probs[0]}')
        # compute new means and variances
        for k in range(self.k):
            # new means
            self.mu[k] = np.sum(np.diag(self.post_probs[:,k]) @ self.data,axis=0) / sum_per_cluster
            # new variances
            self.sigma[k] = np.sum([self.post_probs[u,k] * (self.data-self.mu[k]) @ np.transpose(self.data-self.mu[k]) for u in range(self.n)])
            self.sigma[k] = self.sigma[k] / sum_per_cluster
        print('MAXIMIZE')
        print(f'cl_probs: {self.cl_probs[0]}---\nmu: -----\n{self.mu[0]}\nsigma: ----\n{self.sigma[0]}')
        print('MAXIMIZE DONE')
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
        ax2 = fig.add_subplot(111)
        # calculate probability distribution
        z = np.zeros((gran,gran))
        for k in range(self.k):
            print(f'{k}--------\n{self.sigma[k]}')
            rv = multivariate_normal(self.mu[k], self.sigma[k])
            z += self.cl_probs[k] * rv.pdf(pos)
        # plot heat map and data points
        ax2.contourf(x, y, z)
        ax2.plt(self.data[:,dim1], self.data[:,dim2],'k.')
        # save figure
        plt.savefig(f'img/iter{iteration}.png')

    def clustering(self):
        # Parameters already initialized in init
        iteration = 0
        while self.step():
            self.visualize(iteration)
            iteration = iteration + 1
        return iteration

