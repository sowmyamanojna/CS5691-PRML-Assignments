import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

class GMM():
    def __init__(self, q):
        self.q = q

    def fit(self, X, covariance_type="full", tol=1e-5):
        """
        X: n*d
        mu: q*d
        C: q*d*d
        gamma: n*q
        """
        self.n, self.d = X.shape    
        self.X = X
        self.covariance_type = covariance_type
        self.initialization()
        self.lglk_list = []
        for i in tqdm(range(100)):
            self.lglk_list.append(self.log_likelihood(self.X))
            self.expectation()
            self.maximization()
            new_lk = self.log_likelihood(self.X)
            if new_lk - self.lglk_list[-1] < tol:
                break


    def initialization(self):
        kmeans = KMeans(n_clusters=self.q, random_state=0).fit(self.X)
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)

        self.subcomponents = unique.size
        self.gamma = np.eye(self.subcomponents)[labels]
        self.Nq = np.sum(self.gamma, axis=0)
        self.weights = counts/self.n
        self.mu = (self.gamma.T @ self.X)/self.Nq.reshape(-1,1)
        self.C = np.zeros((self.subcomponents, self.d, self.d))

        for i in range(self.q):
            self.C[i] = (1/self.Nq[i])*(self.gamma[:,i].reshape(-1,1)*(self.X-self.mu[i,:])).T@(self.X-self.mu[i,:])


    def expectation(self):
        self.gamma = np.zeros((self.n, self.q))

        for i in range(self.q):
            self.gamma[:,i] = self.weights[i]*mvn.pdf(self.X, self.mu[i], self.C[i])
        self.gamma = self.gamma/np.sum(self.gamma, axis=1).reshape(-1,1)

    def maximization(self):
        # print(np.sum(self.weights))
        self.Nq = np.sum(self.gamma, axis=0)
        self.mu = (self.gamma.T @ self.X)/self.Nq.reshape(-1,1)
        
        for i in range(self.q):
            self.C[i] = (1/self.Nq[i])*(self.gamma[:,i].reshape(-1,1)*(self.X-self.mu[i,:])).T@(self.X-self.mu[i,:])
        self.weights = self.Nq/self.n

    def log_likelihood(self, X_test):
        lk = 0
        n, d = X_test.shape
        for i in range(n):
            val = 0
            for j in range(self.q):
                val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j])
            lk += np.log(val)

        return lk

    def indv_log_likelihood(self, X_test):
        n, d = X_test.shape
        lk = np.zeros((X_test.shape[0], 1))
        for i in range(n):
            val = 0
            for j in range(self.q):
                val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j])
            lk[i] = np.log(val)

        return lk
