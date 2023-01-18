import numpy as np
from scipy.stats import multivariate_normal
import GMM


def main():
    # generation multivariate data
    mnd1 = multivariate_normal(np.array([0,0]), np.array([[2,0],[0,3]]))
    mnd2 = multivariate_normal(np.array([1,2]), np.array([[1,0.75],[0.75,2]]))
    data = np.concatenate((mnd1.rvs(15),mnd2.rvs(10)),axis=0)

    clustering = GMM.GMM(data, 2, threshold=0.1)


if __name__ == "__main__":
    main()