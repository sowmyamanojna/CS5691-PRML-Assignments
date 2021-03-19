import os
import missingno
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from preprocess import PreProcess
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans

class GaussianBasis():
    def __init__(self):
        pass

    def fit_grid(self, df_new, df_save, num_clusters, regularization=None, lmbda_list=[0], verbose=False, show=False):
        n_clu_hist = []
        lmbda_hist = []
        error1_hist = []
        error2_hist = []
        sse_dict = {}

        if regularization == None or regularization=="L2":
            print("Running K-Means...", end="")
            for n_clu in tqdm(num_clusters):
                kmeans = KMeans(n_clusters=n_clu, random_state=42).fit(df_new.to_numpy())
                sse_dict[n_clu] = kmeans.inertia_

                mean_centers = kmeans.cluster_centers_
                corresponding_center = mean_centers[kmeans.labels_,:]

                X = df_new.to_numpy()
                distance = np.linalg.norm(X-corresponding_center, axis=1)
                var = np.var(distance)*distance.size

                phi = np.ones((X.shape[0], 1))
                for i in range(n_clu):
                    phi = np.append(phi, np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var).reshape(-1,1), axis=1)

                for lmbda in lmbda_list:
                    n_clu_hist.append(n_clu)
                    lmbda_hist.append(lmbda)

                    W1 = (np.linalg.inv(phi.T @ phi + lmbda*np.identity(phi.shape[1])) @ phi.T) @ df_save["Next_Tmin"]
                    W2 = (np.linalg.inv(phi.T @ phi + lmbda*np.identity(phi.shape[1])) @ phi.T) @ df_save["Next_Tmax"]
                    W1 = W1.reshape(-1,1)
                    W1 = W2.reshape(-1,1)
                    pred1 = phi @ W1
                    pred2 = phi @ W2

                    plt.figure(figsize=[16,9])
                    plt.title("Clusters: "+str(n_clu))
                    plt.subplot(1, 2, 1)
                    plt.hist(pred1, alpha=0.5)
                    plt.hist(df_save["Next_Tmin"], alpha=0.5)
                    plt.title("Next_Tmin; Clusters: "+str(n_clu))
                    plt.grid()
                    plt.subplot(1, 2, 2)
                    plt.plot(df_save["Next_Tmin"], pred1, ".")
                    plt.plot(df_save["Next_Tmin"], df_save["Next_Tmin"], '.')
                    plt.title("Next_Tmin; Clusters: "+str(n_clu))
                    plt.grid()
                    if show:
                        plt.show()
                    fname = "fit_1_k_"+str(num_clusters)+"_lmbda_"+str(lmbda)+".png"
                    plt.savefig("images/q3/"+fname)

                    plt.figure(figsize=[16,9])
                    plt.title("Clusters: "+str(n_clu))
                    plt.subplot(1, 2, 1)
                    plt.hist(pred2, alpha=0.5, label="Predicted")
                    plt.hist(df_save["Next_Tmax"], alpha=0.5, label="True")
                    plt.title("Next_Tmax; Clusters: "+str(n_clu))
                    plt.grid()
                    plt.subplot(1, 2, 2)
                    plt.plot(df_save["Next_Tmax"], pred2, ".")
                    plt.plot(df_save["Next_Tmax"], df_save["Next_Tmax"], '.')
                    plt.title("Clusters: "+str(n_clu))
                    plt.grid()
                    if show:
                        plt.show()
                    fname = "fit_2_k_"+str(num_clusters)+"_lmbda_"+str(lmbda)+".png"
                    plt.savefig("images/q3/"+fname)

                    error1 = np.linalg.norm(df_save["Next_Tmin"].to_numpy().reshape(-1,1)-pred1)
                    error2 = np.linalg.norm(df_save["Next_Tmin"].to_numpy().reshape(-1,1)-pred2)

                    error1_hist.append(error1)
                    error2_hist.append(error2)
                print("Done!")

        df_result = pd.DataFrame()
        df_result["Cluster"] = n_clu_hist
        df_result["Lambda"] = lmbda_hist
        df_result["Error 1"] = error1_hist
        df_result["Error 2"] = error2_hist
        df_result.to_csv("../datasets/q3_gridsearch.csv")
        
        return df_result, sse_dict