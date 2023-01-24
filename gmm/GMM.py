import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from visualize import visualize_cont
import csv

class GMM:

    def __init__(self, data, k, threshold=0.1, max_iter=1000, probs=None, labels=None):
        # initialize logging
        self.file = open(f'logs_{k}/params.csv', 'w')
        self.writer = csv.writer(self.file)

        # initializing basic variables
        self.threshold = threshold
        self.max_iter = max_iter
        self.k = k
        self.data = data
        self.labels = labels
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
        sigmas = []
        for k in range(self.k):
            sigmas.append(np.expand_dims(np.diag((self.maxdims[:,1]-self.maxdims[:,0])/self.k), axis=0))
        # rng.random(self.dim)) made posterior probabilities zero
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
        mnd = multivariate_normal(self.mu[k], self.sigma[k],allow_singular=True)
        return mnd.pdf(x) # probability density function

    def estimate(self):
        # Computes posterior probabilities for unit u to belong to cluster i for all u,i
        old_post_prob = np.copy(self.post_probs)
        # update posterior probabilities
        for u in range(self.n):
            comp_dens = np.array([self.density_scipy(i,self.data[u]) for i in range(self.k)]) # shape (k,)
            divisor = np.dot(comp_dens, self.cl_probs)
            if divisor == 0.:
                divisor = 1
                print("DIVISOR WAS ZERO")
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

    def clustering(self):
        # Parameters already initialized in init
        iteration = 0
        #self.visualize_cont(gmm=self, iteration=-1, probs=False)
        #self.visualize_cont(gmm=self, iteration=iteration)
        self.log_params(iteration)
        while self.step():
            iteration = iteration + 1
            self.log_params(iteration=iteration)
            #self.visualize_cont(gmm=self, iteration=iteration)
            print(f'Iteration {iteration}')
            if iteration==self.max_iter:
                print('Reached maximum number of iterations')
                break
        self.log_pis()
        self.file.close()
        return iteration

    def log_pis(self):
        with open(f'logs_{self.k}/pis.csv','w') as file:
            writer = csv.writer(file)
            row = ['day']
            row += list(map(lambda s: 'cluster'+str(s), range(self.k)))
            row += ['weighted_mean', 'max']
            writer.writerow(row)
            for i, point in enumerate(self.post_probs):
                row = [i] + list(self.post_probs[i])
                row += [np.inner(self.post_probs[i], np.arange(0,self.k))] # mean
                row += [np.argmax(self.post_probs[i])] # max
                writer.writerow(row)
        return

    def log_params(self, iteration):
        if iteration == 0:
            row = ['day','cluster',]
            row += list(map(lambda s: 'cluster'+str(s), range(self.k)))
            row += list(map(lambda la: la+'_mu', self.labels))
            idxs = np.concatenate([[str(i)+str(j) for j in range(self.dim)] for i in range(self.dim)])
            row += list(map(lambda i: 'sigma'+i, idxs))
            self.writer.writerow(row)
        for k in range(self.k):
            row = [int(iteration),int(k)] + list(self.cl_probs) + list(self.mu[k]) + list(self.sigma[k].flatten())
            self.writer.writerow(row)

