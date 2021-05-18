import numpy as np

class PolynomialRegression():
    def __init__(self):
        pass

    def fit(self, X, y, degree=2, lmbda=0):
        self.degree = degree
        self.lmbda = lmbda

        X_poly = self.get_polynomial_features(X)
        self.get_weights(X_poly, y)
        return X_poly

    def transform(self, X_val):
        X_poly = self.get_polynomial_features(X_val)
        y_val = X_poly @ self.W
        return y_val

    def fit_transform(self, X, y, degree=2, lmbda=0):
        self.fit(X, y, degree, lmbda)
        return self.transform(X)

    def get_polynomial_features(self, X):
        X_new = np.ones(X.shape)
        for i in range(1, self.degree+1):
            X_new = np.append(X_new, X**i, axis=1)
        return X_new

    def get_weights(self, X_poly, y):
        d = X_poly.shape[1]
        self.W = ((np.linalg.inv(X_poly.T @ X_poly + self.lmbda*np.identity(d))) @ X_poly.T) @ y

    def error(self, y_true, y_pred):
        rmse = np.linalg.norm(y_pred-y_true)/(y_true.size)**0.5
        return rmse
