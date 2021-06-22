import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

class GMM():
    """
    The difference between GMM and GMM_2B
    is the initialization.
    GMM uses K-Means
    GMM_2B uses random init
    """
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
            new_lk = self.log_likelihood_new(self.X)
            diff = new_lk - self.lglk_list[-1]
            if diff > 0:
                self.update()
            if  diff < tol:
                # if diff < 0: print("Difference is less than 0")
                break

    def update(self):
        self.C = self.C_new.copy()
        self.gamma = self.gamma_new.copy()
        self.weights = self.weights_new.copy()
        self.Nq = self.Nq_new.copy()
        self.mu = self.mu_new.copy()

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

        self.C_new = self.C.copy()


    def expectation(self):
        self.gamma_new = np.zeros((self.n, self.q))
        for i in range(self.q):
            try:
                self.gamma_new[:,i] = self.weights[i]*mvn.pdf(self.X, self.mu[i], self.C[i])
            except:
                self.gamma_new[:,i] = self.weights[i]*mvn.pdf(self.X, self.mu[i], self.C[i]+np.eye(self.C[i].shape[0])*1e-7)
        self.gamma_new = self.gamma_new/np.sum(self.gamma_new, axis=1).reshape(-1,1)

    def maximization(self):
        # print(np.sum(self.weights))
        self.Nq_new = np.sum(self.gamma_new, axis=0)
        self.mu_new = (self.gamma_new.T @ self.X)/self.Nq_new.reshape(-1,1)
        
        for i in range(self.q):
            self.C_new[i] = (1/self.Nq_new[i])*(self.gamma_new[:,i].reshape(-1,1)*(self.X-self.mu_new[i,:])).T@(self.X-self.mu_new[i,:])

            if self.covariance_type == "diag":
                self.C_new[i] = np.diag(self.C_new[i])

        self.weights_new = self.Nq_new/self.n

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

    def log_likelihood_new(self, X_test):
        lk = 0
        n, d = X_test.shape
        for i in range(n):
            val = 0
            for j in range(self.q):
                try:
                    val += self.weights_new[j]*mvn.pdf(X_test[i], self.mu_new[j], self.C_new[j])
                except:
                    val += self.weights_new[j]*mvn.pdf(X_test[i], self.mu_new[j], self.C_new[j]+np.eye(self.C_new[j].shape[0])*1e-7)
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

class GMM_2A():
    def __init__(self, q, tol=1e-3):
        self.q = q
        self.tol = tol

    def fit(self, X, covariance_type="full", epochs=100, tol=1e-6):
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
        self.tol = tol
        self.initialization()
        self.lglk_list = []

        for _ in range(self.epochs):
            self.lglk_list.append(self.log_likelihood(self.X))
            self.expectation()
            self.maximization()

    def initialization(self):
        self.C = 5*np.zeros((self.q, self.d, self.d))
        for i in range(self.q):
            np.fill_diagonal(self.C[i], 5)

        self.mu = np.random.randint(0, 1, size=(self.q, self.d))
        self.weights = np.ones((self.q, 1))/self.q
        self.gamma = np.zeros((self.n, self.q))
        self.reg_C = self.tol*np.eye(self.d)

    def expectation(self):
        self.gamma = np.zeros((self.n, self.q))
        # Update the gamma
        for i in range(self.q):
            self.gamma[:,i] = self.weights[i]*mvn.pdf(self.X, self.mu[i], self.C[i]+self.reg_C)
        # Normalize the gamma
        self.gamma = self.gamma/np.sum(self.gamma, axis=1).reshape(-1,1)

    def maximization(self):
        self.Nq = np.sum(self.gamma, axis=0)
        self.mu = (self.gamma.T @ self.X)/self.Nq.reshape(-1,1)
        
        for i in range(self.q):
            self.C[i] = (1/self.Nq[i])*(self.gamma[:,i].reshape(-1,1)*(self.X-self.mu[i,:])).T@(self.X-self.mu[i,:])
            self.C[i] += self.reg_C

            if self.covariance_type == "diag":
                self.C[i] = np.diag(self.C[i])

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
                try:
                    val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j])
                except:
                    val += self.weights[j]*mvn.pdf(X_test[i], self.mu[j], self.C[j]+np.eye(self.C[j].shape[0])*1e-7)
            lk[i] = np.log(val)

        return lk

    def mvg(self, X, m, C):
        n, d = X.shape
        denom = (1/(((2*np.pi)**(d/2))*np.sqrt(np.linalg.det(C))))
        exp = (-(X-m).T@np.linalg.pinv(C)@(X-m))/2
        return denom*np.exp(exp)

    def gaussian_val(self, X_test):
        n, d = X_test.shape
        val = np.zeros((n, self.q))

        for i in range(self.q):
            val[:,i] = self.weights[i]*mvn.pdf(X_test, self.mu[i], self.C[i])

        self.val = val
        return np.sum(val, axis=1)

class GMM_2B():
    def __init__(self, q, tol=1e-3):
        self.q = q
        self.tol = tol

    def fit(self, X, diag=False, epochs=100, tol=1e-6):
        """
        X: n*d
        mu: q*d
        C: q*d*d
        gamma: n*q
        """
        self.n, self.d = X.shape    
        self.X = X
        self.diag = diag
        self.epochs = epochs
        self.tol = tol
        
        self.initialization()

        self.lglk_list = []

        for i in range(self.epochs):
            self.lglk_list.append(self.log_likelihood(self.X))
            self.expectation()
            self.maximization()

    def initialization(self):
        self.C = 5*np.zeros((self.q, self.d, self.d))
        for i in range(self.q):
            np.fill_diagonal(self.C[i], 5)

        self.mu = np.random.randint(-1, 1, size=(self.q, self.d))
        self.weights = np.ones((self.q, 1))/self.q
        self.gamma = np.zeros((self.n, self.q))
        self.reg_C = self.tol*np.eye(self.d)

    def expectation(self):
        self.gamma = np.zeros((self.n, self.q))
        # Update the gamma
        for i in range(self.q):
            self.gamma[:,i] = self.weights[i]*mvn.pdf(self.X, self.mu[i], self.C[i]+self.reg_C)
        # Normalize the gamma
        self.gamma = self.gamma/np.sum(self.gamma, axis=1).reshape(-1,1)

    def maximization(self):
        self.Nq = np.sum(self.gamma, axis=0)
        self.mu = (self.gamma.T @ self.X)/self.Nq.reshape(-1,1)
        
        for i in range(self.q):
            self.C[i] = (1/self.Nq[i])*(self.gamma[:,i].reshape(-1,1)*(self.X-self.mu[i,:])).T@(self.X-self.mu[i,:])
            self.C[i] += self.reg_C

            if self.diag == True:
                # np.diag alone returns an array of diagonal values
                # The outer np.diag is necessary to make it a diagonal
                # matrix again.
                self.C[i] = np.diag(np.diag(self.C[i]))

        self.weights = self.Nq/self.n

    def log_likelihood(self, X_test):
        lk = 0
        n, d = X_test.shape
        for i in range(n):
            val = 0
            for j in range(self.q):
                # print(self.C[j])
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

    def mvg(self, X, m, C):
        n, d = X.shape
        denom = (1/(((2*np.pi)**(d/2))*np.sqrt(np.linalg.det(C))))
        exp = (-(X-m).T@np.linalg.pinv(C)@(X-m))/2
        return denom*np.exp(exp)

    def gaussian_val(self, X_test):
        n, d = X_test.shape
        val = np.zeros((n, self.q))

        for i in range(self.q):
            val[:,i] = self.weights[i]*mvn.pdf(X_test, self.mu[i], self.C[i])
        self.val = val
        return np.sum(val, axis=1)

class GMM_vl():
    def __init__(self, q, tol=1e-3):
        self.q = q
        self.tol = tol
        self.new_probab = {}

    def fit(self, class_name, epochs=100, diag=False):
        self.epochs = epochs
        self.class_name = class_name

        class_path = "../datasets/2B/" + class_name + "/"
        file_names = os.listdir(class_path)

        train_file_names = [i for i in file_names if "csv" in i and "train_" in i]
        train_file_names.sort()

        # Training files
        self.gmm_list = []
        self.probab_list = []

        for i in train_file_names:
            gmm = GMM_2B(self.q)

            file_path = class_path + i
            df = pd.read_csv(file_path, header=None)
            X = df.to_numpy()

            gmm.fit(X, epochs=epochs, diag=diag)
            probab = gmm.gaussian_val(X)

            self.gmm_list.append(gmm)
            self.probab_list.append(probab)

    def get_probability(self, class_name, dev=False):
        class_path = "../datasets/2B/" + class_name + "/"
        file_names = os.listdir(class_path)

        train_file_names = [i for i in file_names if "csv" in i and "train_" in i]
        train_file_names.sort()
        dev_file_names = [i for i in file_names if "csv" in i and "dev_" in i]
        dev_file_names.sort()

        if dev == False:
            choosen_file_names = train_file_names
        else:
            choosen_file_names = dev_file_names

        self.new_probab[class_name] = []
        for i, j in enumerate(choosen_file_names):
            gmm = self.gmm_list[i]

            file_path = class_path + j
            df = pd.read_csv(file_path, header=None)
            X = df.to_numpy()
            probab = gmm.gaussian_val(X)
            self.new_probab[class_name].append(probab)
        
        self.new_probab[class_name] = np.array(self.new_probab[class_name])

class GMM_vl_classifier():
    def __init__(self):
        pass

    def compile(self, class_wise_gmm):
        self.class_wise_gmm = class_wise_gmm
        self.gmm_class_names = list(class_wise_gmm.keys())

    def transform(self, class_names="", dev=False):
        if class_names == "":
            class_names = self.gmm_class_names

        # Loop over all classes
        for gmm_class_name in self.gmm_class_names:
            gmm_vl = self.class_wise_gmm[gmm_class_name]

            # Get the probab for all the data points for a particular class
            for data_class_name in class_names:
                gmm_vl.get_probability(data_class_name, dev=dev)

    def classify(self, dev=False, eps=1e-6):
        self.transform(dev=dev)

        self.test = {}
        for i in self.gmm_class_names:
            """
            Get probabilities of the size 
            n*36*number of classes
            n * d * m
            """
            d = self.class_wise_gmm[i].new_probab[i].shape[0]
            n = self.class_wise_gmm[i].new_probab[i].shape[1]
            m = len(self.gmm_class_names)
            self.test[i] = np.zeros((n, d, m))

        for k, i in enumerate(self.class_wise_gmm):
            """
            Each of these class wise gmms have a new probab.
            The sizes of these new probabs are:
            36*n1
            36*n2 and so on

            Hence, for each dataset, the values across these
            gmms should be considered.
            """
            self.class_wise_gmm[i].probab_vec = {}
            for j in self.class_wise_gmm[i].new_probab:
                val = self.class_wise_gmm[i].new_probab[j].T
                self.test[j][:,:,k] = val
                self.class_wise_gmm[i].probab_vec[j] = val

        self.predictions = {}
        self.accuracy = {}
        for k, i in enumerate(self.gmm_class_names):
            # Get all predictions made for that classes' data
            a = self.test[i]
            # Normalize across classes
            b = a/np.sum(a, axis=2).reshape(a.shape[0], a.shape[1], 1)
            # Get product across the 36 dimensions
            c = np.prod(b, axis=1)
            class_predicted = np.argmax(c, axis=1)
            self.predictions[i] = class_predicted
            self.accuracy[i] = 100*(np.sum(class_predicted==k)/class_predicted.size)