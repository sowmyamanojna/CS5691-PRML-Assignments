################################################################### [markdown]
# ## CS5691 PRML Assignment 1
# **Team 1**  
# **Team Members:**  
# N Sowmya Manojna   BE17B007  
# Thakkar Riya Anandbhai  PH17B010   
# Chaithanya Krishna Moorthy  PH17B011   

###################################################################
# Install required Packages
# Uncomment if you are running for the firts time
# !pip install -r requirements.txt
# try:
#     !mkdir images
# except:
#     pass

###################################################################
import numpy as np
np.random.seed(0)
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.style.use('science')
plt.rcParams['font.size'] = 18
plt.rcParams['axes.grid'] = True
plt.rcParams["grid.linestyle"] = (5,9)
plt.rcParams['figure.figsize'] = 8,6

###################################################################
df = pd.read_csv("../datasets/function1.csv", index_col=0)
df.sort_values(by=["x"], inplace=True)
df.head()

###################################################################
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

class GridSearch():
    def __init__(self):
        pass

    def get_result(self, df, sample_size, degrees_allowed, lmbda_list):
        df_sample = df.sample(n=sample_size, random_state=42)

        df_train = df_sample.sample(frac=0.9, random_state=42)
        df_val = df_sample[~df_sample.index.isin(df_train.index)]

        X_train = df_train["x"].to_numpy().reshape(-1, 1)
        X_val = df_val["x"].to_numpy().reshape(-1, 1)

        y_train = df_train["y"].to_numpy().reshape(-1, 1)
        y_val = df_val["y"].to_numpy().reshape(-1, 1)
        
        self.result = []
        self.correspondance = {}
        for degree in degrees_allowed:
            for lmbda in lmbda_list:
                regressor = PolynomialRegression()
                regressor.fit(X_train, y_train, degree=degree, lmbda=lmbda)
                y_train_pred = regressor.transform(X_train)
                y_val_pred = regressor.transform(X_val)

                train_error = regressor.error(y_train, y_train_pred)
                val_error = regressor.error(y_val, y_val_pred)

                self.result.append([degree, lmbda, train_error, val_error])
                self.correspondance[(degree, lmbda)] = {"df_sample":df_sample, "regressor":regressor}

        df_results = pd.DataFrame(self.result, columns=["degree", "lambda", "Train error", "Validation error"])
        df_results["Sum Error"] = df_results["Train error"] + df_results["Validation error"]
        df_results.sort_values(by="Sum Error", inplace=True)
        return df_results, self.correspondance

    def get_plots(self, X, y, sample_size, show):
        for key in correspondance:
            df_sample = correspondance[key]["df_sample"]
            df_sample.sort_values(by=["x"], inplace=True)
            X_sample = df_sample["x"].to_numpy().reshape(-1,1)
            y_sample = df_sample["y"].to_numpy().reshape(-1,1)

            regressor = correspondance[key]["regressor"]
            y_pred_sample = regressor.transform(X_sample)

            title = "Curve Fitting - Degree: "+str(regressor.degree) \
            +"; Sample Size: "+str(sample_size)+"; $\lambda$: " \
            +str(regressor.lmbda)
            fname = "d_"+str(regressor.degree)+"_size_"+str(sample_size)+"_l_"+str(regressor.lmbda)+".png"
            
            plt.figure()
            plt.plot(X, y, label="True Value")
            if y_sample.size >= 100:
                plt.plot(X_sample, y_sample, 'r.', alpha=0.5, label="Sampled points")
            else:
                plt.plot(X_sample, y_sample, 'ro', alpha=0.75, label="Sampled points")
            plt.plot(X_sample, y_pred_sample, label="Predicted Value")
            if title:
                plt.title(title)
            plt.xlabel("X-values")
            plt.ylabel("Y-values")
            plt.legend()
            plt.savefig("images/"+fname)
            if show:
                plt.show()
###################################################################
lmbda_list = [0, 0.5, 1, 2, 10, 50, 100]
degrees_allowed = [2, 3, 6, 9]
datasizes_considered = [10, 200]
complete_dataset_size = df.shape[0]

X = df["x"].to_numpy().reshape(-1,1)
y = df["y"].to_numpy().reshape(-1,1)

results_df_list = []
correspondance_list = []
for sample_size in datasizes_considered:
    gridsearch = GridSearch()
    df_result, correspondance = gridsearch.get_result(df, sample_size=sample_size, degrees_allowed=degrees_allowed, lmbda_list=lmbda_list)
    results_df_list.append(df_result)
    correspondance_list.append(correspondance)

    print("\nFor Sample Size of ", sample_size, " - GridSearch Results:")
    print(df_result)
    print("="*70)
    gridsearch.get_plots(X, y, sample_size, show=False)
###################################################################

# From the resuts obtained, we see that degree=6, lambda=0.0 
# best fits the model.

###################################################################

best_degree = int(df_result.iloc[0]["degree"])
best_lmbda = df_result.iloc[0]["lambda"]

df_train = df.sample(frac=0.7, random_state=42)
df_test = df[~df.index.isin(df_train.index)]

X_train = df_train["x"].to_numpy().reshape(-1,1)
X_test = df_test["x"].to_numpy().reshape(-1,1)
y_train = df_train["y"].to_numpy().reshape(-1,1)
y_test = df_test["y"].to_numpy().reshape(-1,1)

regressor = PolynomialRegression()
X_train_poly = regressor.fit(X_train, y_train, degree=best_degree, lmbda=best_lmbda)
y_train_pred = regressor.transform(X_train)
y_test_pred = regressor.transform(X_test)
train_error = regressor.error(y_train, y_train_pred)
test_error = regressor.error(y_test, y_test_pred)

print("Training Error:", train_error)
print("Testing Error:", test_error)
