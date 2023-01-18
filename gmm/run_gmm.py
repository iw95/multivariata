import numpy as np
from scipy.stats import multivariate_normal
import GMM
import kmeans


def normal_data():
    # generating multivariate data
    mnd1 = multivariate_normal(np.array([0, 0]), np.array([[2, 0], [0, 3]]))
    mnd2 = multivariate_normal(np.array([1, 2]), np.array([[1, 0.75], [0.75, 2]]))
    data = np.concatenate((mnd1.rvs(15), mnd2.rvs(10)), axis=0)
    return data


def gmm():
    data = normal_data()
    clustering = GMM.GMM(data, 2, threshold=0.1)


def kmeaning():
    data = normal_data()
    km = kmeans.Kmeans(data, 2, iterations=3)

def gmm_on_kmeans():
    k = 2
    data = normal_data()
    km = kmeans.Kmeans(data,k,iterations=3)
    probs = np.concatenate([np.expand_dims(np.array(km.labels==i, dtype=float),axis=1) for i in range(k)],axis=1)
    gmm = GMM.GMM(data, k=k, threshold=0.1, probs=probs)

def main():
    gmm_on_kmeans()


if __name__ == "__main__":
    main()