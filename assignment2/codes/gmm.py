import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

class GMM():
    def __init__(self, q, tol=1e-3):
        self.q = q
        self.tol = tol

    def fit(self, X, covariance_type="full", epochs=100, tol=1e-5):
        """
        X: n*d
        mu: q*d
        C: q*d*d
        gamma: n*q
        """
        self.n, self.d = X.shape    
        self.X = X
        self.epochs = epochs
        self.covariance_type = covariance_type
        self.initialization()
        self.lglk_list = []
        for i in range(self.epochs):
            self.lglk_list.append(self.log_likelihood(self.X))
            self.expectation()
            self.maximization()
            new_lk = self.log_likelihood(self.X)
            diff = new_lk - self.lglk_list[-1]
            if  diff < tol:
                # if diff < 0: print("Difference is less than 0")
                break

    def initialization(self):
        # kmeans = KMeans(n_clusters=self.q, random_state=0).fit(self.X)
        kmeans = KMeans(n_clusters=self.q).fit(self.X)
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

            if self.covariance_type == "diag":
                self.C[i] = np.diag(self.C[i])


    def expectation(self):
        self.gamma = np.zeros((self.n, self.q))

        for i in range(self.q):
            try:
                self.gamma[:,i] = self.weights[i]*mvn.pdf(self.X, self.mu[i], self.C[i])
            except:
                self.gamma[:,i] = self.weights[i]*mvn.pdf(self.X, self.mu[i], self.C[i]+np.eye(self.C[i].shape[0])*1e-7)
        self.gamma = self.gamma/np.sum(self.gamma, axis=1).reshape(-1,1)

    def maximization(self):
        # print(np.sum(self.weights))
        self.Nq = np.sum(self.gamma, axis=0)
        self.mu = (self.gamma.T @ self.X)/self.Nq.reshape(-1,1)
        
        for i in range(self.q):
            self.C[i] = (1/self.Nq[i])*(self.gamma[:,i].reshape(-1,1)*(self.X-self.mu[i,:])).T@(self.X-self.mu[i,:])

            if self.covariance_type == "diag":
                self.C[i] = np.diag(self.C[i])

        self.weights = self.Nq/self.n

    def log_likelihood(self, X_test):
        lk = 0
        n, d = X_test.shape
        for i in range(n):
            val = 0
            for j in range(self.q):
                try:
                    val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j])
                except:
                    val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j]+np.eye(self.C[j].shape[0])*1e-7)
            lk += np.log(val)

        return lk

    def indv_log_likelihood(self, X_test):
        n, d = X_test.shape
        lk = np.zeros((X_test.shape[0], 1))
        for i in range(n):
            val = 0
            for j in range(self.q):
                try:
                    val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j])
                except:
                    val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j]+np.eye(self.C[j].shape[0])*1e-7)
            lk[i] = np.log(val)

        return lk

    def mvg(self, X, m, C):
        n, d = X.shape
        print(X.shape, m.shape, C.shape, (X-m).shape)
        denom = (1/(((2*np.pi)**(d/2))*np.sqrt(np.linalg.det(C))))
        exp = (-(X-m).T@np.linalg.pinv(C)@(X-m))/2
        return denom*np.exp(exp)

    def gaussian_val(self, X_test):
        n, d = X_test.shape
        val = np.zeros((n, self.q))

        for i in range(self.q):
            try:
                val[:,i] = self.weights[i]*mvn.pdf(X_test, self.mu[i], self.C[i], allow_singular=True)
            except:
                self.C[i] += self.tol
                val[:,i] = self.weights[i]*self.mvg(X_test, self.mu[i], self.C[i], allow_singular=True)

        return np.sum(val, axis=1)

class GMM_vl():
    def __init__(self, q, tol=1e-3):
        self.q = q
        self.tol = tol

    def fit(self, class_name, epochs=100):
        self.epochs = epochs
        self.class_name = class_name

        class_path = "../datasets/2B/" + class_name + "/"
        file_names = os.listdir(class_path)

        train_file_names = [i for i in file_names if "csv" in i and "train_" in i]
        dev_file_names = [i for i in file_names if "csv" in i and "dev_" in i]

        # Training files
        self.gmm_list = []
        self.probab_list = []

        for i in tqdm(train_file_names):
            gmm = GMM(self.q)

            file_path = class_path + i
            df = pd.read_csv(file_path, header=None)
            X = df.to_numpy()

            gmm.fit(X, epochs=epochs)
            probab = gmm.gaussian_val(X)

            self.gmm_list.append(gmm)
            self.probab_list.append(probab)

    def get_probability(self, class_name):
        class_path = "../datasets/2B/" + class_name + "/"
        file_names = os.listdir(class_path)

        train_file_names = [i for i in file_names if "csv" in i and "train_" in i]
        dev_file_names = [i for i in file_names if "csv" in i and "dev_" in i]
        
        size_list = []
        self.new_probab = []
        for i, j in enumerate(tqdm(train_file_names)):
            gmm = self.gmm_list[i]

            file_path = class_path + j
            df = pd.read_csv(file_path, header=None)
            X = df.to_numpy()

            size_list.append(X.shape[0])
            probab = gmm.gaussian_val(X)
            print(j, probab.shape[0], size_list[-1], probab.shape[0] == size_list[-1])
            assert probab.shape[0] == size_list[-1]

            self.new_probab.append(probab)

    def get_log_likelihood(self, X_test):
        value = np.ones((X_test.shape[0], 1))
        for gmm in self.gmm_list:
            value *= gmm.indv_log_likelihood(X_test)

        return value
