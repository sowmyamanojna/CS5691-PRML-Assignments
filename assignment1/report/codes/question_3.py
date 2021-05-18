###################################################################
# ## CS5691 PRML Assignment 1
# **Team 1**  
# **Team Members:**  
# N Sowmya Manojna   BE17B007  
# Thakkar Riya Anandbhai  PH17B010   
# Chaithanya Krishna Moorthy  PH17B011   

###################################################################
# Install required Packages
# Uncomment if you are running for the first time
# !pip install -r requirements.txt
# try:
#     !mkdir images/q3
# except:
#     pass

###################################################################
import os
import missingno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import PreProcess
from gaussianbasis import GaussianBasis
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.stats.outliers_influence import variance_inflation_factor

###################################################################
print("Reading Dataset... ", end="")
df = pd.read_csv("../datasets/1_bias_clean.csv")
print("Done!")

print("Starting Preprocessing... ")
preprocess = PreProcess()
preprocess.clean(df, verbose=True)
print("="*60)
print("="*60)
print("Preprocessing Done!!")

df_save = pd.read_csv("../datasets/processed.csv", index_col=0)
df_new = df_save.copy()
df_new = df_new.drop(["Next_Tmax", "Next_Tmin"], axis=1)

num_clusters = [1]
num_clusters.extend(range(2,10))
num_clusters.extend(range(15, 31, 5))
num_clusters.extend(range(40, 101, 10))

print("Starting Regularization... ", end="")
lmbda_list = [0, 0.5, 1, 2, 10, 50, 100]
regressor = GaussianBasis()
output = regressor.fit_grid(df_new, df_save, num_clusters, regularization="L2", lmbda_list=lmbda_list, verbose=True, show=False)
df_result, sse_dict, num_clusters = output


df_result["Sum Error"] = df_result["Error 1"] + df_result["Error 2"]
df_result.sort_values(by=["Sum Error"], inplace=True)
print(df_result)

sse_list = [sse_dict[i] for i in sse_dict]
plt.figure(figsize=[12,8])
plt.plot(num_clusters,sse_list )
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Knee Plot for determining the number of clusters")
plt.grid()
plt.show()

plt.figure(figsize=[12,8])
plt.plot(num_clusters[:5], sse_list[:5])
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Knee Plot for determining the number of clusters")
plt.grid()
plt.show()