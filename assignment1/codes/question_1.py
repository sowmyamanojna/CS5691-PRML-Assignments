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
#     !mkdir images/q1
# except:
#     pass

###################################################################
import numpy as np
np.random.seed(0)
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = 8,6
plt.rcParams['font.serif'] = "Cambria"
plt.rcParams['font.family'] = "serif"

from regression import PolynomialRegression
from gridsearch import GridSearch

###################################################################
df = pd.read_csv("../datasets/function1.csv", index_col=0)
df.sort_values(by=["x"], inplace=True)
df.head()

###################################################################
lmbdas_ = [0, 0.5, 1, 2, 10, 50, 100]
degrees_ = [2, 3, 6, 9]
datasizes_considered = [10, 200]
complete_dataset_size = df.shape[0]

X = df["x"].to_numpy().reshape(-1,1)
y = df["y"].to_numpy().reshape(-1,1)

results_df_list = []
correspondance_list = []

# Loop over each sample size
for sample_size in datasizes_considered:
    # Fit across all degree, lambdas
    gridsearch = GridSearch()
    result = gridsearch.fit(df, sample_size=sample_size, degrees_=degrees_, lmbdas_=lmbdas_)

    df_result, correspondance = result
    results_df_list.append(df_result)
    correspondance_list.append(correspondance)

    print("\nFor Sample Size of ", sample_size, " - GridSearch Results:")
    print(df_result)
    print("="*70)

    gridsearch.plot(X, y, correspondance, sample_size, show=False)
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
