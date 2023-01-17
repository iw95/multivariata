import numpy as np
from scipy.stats import multivariate_normal

class GMM:

    def __init__(self, data, k):
        self.k = k
        self.data = data
        shape = data.shape
        self.n = shape[0]
        self.dim = shape[1]
        self.post_probs = np.zeros((self.n,k))
        self.cl_probs = np.zeros((k))
        self.mu, self.sigma = self.init_params()
        self.det = self.update_det(init=True, det=np.zeros(k))

    def init_params(self):
        return self.rand_init()

    def rand_init(self):
        return (np.random.rand(self.k), np.random.rand(self.k,self.dim,self.dim))

    def update_det(self, init=False, det=None):
        if init == False:
            det = self.det
        for i in range(self.k):
            det[i] = np.linalg.det(self.sigma[i,:,:])
        return det

    def density(self,k,x): # not using a library
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
            comp_dens = np.array([self.density(i,self.data[u]) for i in range(self.k)])
            divisor = np.expand_dims(comp_dens,axis=0) @ np.expand_dims(self.cl_probs,axis=1)
            for i in range(self.k):
                self.post_probs[u,i] = comp_dens[i]*self.cl_probs[i] / divisor

        return False in old_post_prob == self.post_probs # True if at least one entry was changed

    def maximize(self):
        self.cl_probs = np.sum(self.post_probs,axis=0) / self.n
        pass

