#!/usr/bin/env python
# coding: utf-8

# # Assignment 3 - 2 (MLFFNN)
# 
# Team members:
# - N Sowmya Manojna (BE17B007)
# - Thakkar Riya Anandbhai (PH17B010)
# - Chaithanya Krishna Moorthy (PH17B011)

# ## Import Essential Libraries

# In[1]:


import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from ast import literal_eval
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = 12,8
plt.rcParams['font.serif'] = "Cambria"
plt.rcParams['font.family'] = "serif"

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import warnings
warnings.filterwarnings("ignore")

from gridsearch import GridSearch2A


# ## Reading the data, Splitting it

# In[2]:


# Get the data
df = pd.read_csv("../datasets/2A/train_new.csv")
df_test = pd.read_csv("../datasets/2A/dev_new.csv")
display(df.head())

# Split dev into test and validation
df_val, df_test = train_test_split(df_test, test_size=0.3, random_state=42)
display(df_val.head())
display(df_test.head())


# In[3]:


X_train = df.drop("class", axis=1)
y_train = df["class"].to_numpy().astype("int")

X_val = df_val.drop("class", axis=1)
y_val = df_val["class"].to_numpy().astype("int")

X_test = df_test.drop("class", axis=1)
y_test = df_test["class"].to_numpy().astype("int")


# In[4]:


display(df.describe())
display(df_val.describe())
display(df_test.describe())


# ## Preprocessing Dataset

# In[5]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

display(X_train_scaled.describe())
display(X_val_scaled.describe())
display(X_test_scaled.describe())


# ## Training the Model

# In[6]:


parameters = {
              "pca__n_components":list(range(1,25)),
              "mlp__hidden_layer_sizes":[(10,10), (25,25), (50,50), (75,75)], \
              "mlp__batch_size":[50, 100, "auto"], "mlp__alpha":[0.01, 0.001], \
              "mlp__learning_rate":["constant", "adaptive", "invscaling"], \
             }

model = Pipeline([('pca', PCA()), ('mlp', MLPClassifier(max_iter=500, random_state=1))])

clf = GridSearch2A(model, parameters, verbose=1)
clf.fit(X_train, y_train, X_val, y_val)
result_df = pd.DataFrame(clf.cv_results_)
result_df.to_csv("../parameter_search/2A_MLFFNN_train_val.csv")
display(result_df.head(10))


# In[7]:


clf.cv_results_ = clf.cv_results_.sort_values(by=["val_accuracy", "accuracy", "sum_accuracy", "t_inv"], ascending=False, ignore_index=True)

clf.best_params_ = clf.cv_results_.iloc[0].to_dict()
del clf.best_params_["accuracy"]
del clf.best_params_["val_accuracy"]
del clf.best_params_["sum_accuracy"]
del clf.best_params_["t_inv"]


# In[8]:


print("Best Parameters Choosen:")
for i in clf.best_params_:
    print("  - ", i, ": ", clf.best_params_[i], sep="")

pca_params = {}
pca_params["n_components"] = clf.best_params_["n_components"]
mlp_params = clf.best_params_
mlp_params["hidden_layer_sizes"] = literal_eval(mlp_params["hidden_layer_sizes"])
try:
    mlp_params["batch_size"] = int(mlp_params["batch_size"])
except:
    pass

del mlp_params["n_components"]

best_model = Pipeline([('pca', PCA(**pca_params)),                        ('mlp', MLPClassifier(max_iter=500, random_state=1, **mlp_params))])
best_model.fit(X_train, y_train)


# In[9]:


y_pred = best_model.predict(X_train)
print("Accuracy:", 100*np.sum(y_pred==y_train)/y_train.size)
conf_mat = confusion_matrix(y_train, y_pred)
plt.figure()
sns.heatmap(conf_mat, annot=True)
plt.title("2A - Train Confusion Matrix (MLFFNN)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/2A_MLFFNN_train_confmat.png")
plt.show()

y_val_pred = best_model.predict(X_val)
print("Validation Accuracy:", 100*np.sum(y_val_pred==y_val)/y_val.size)
val_conf_mat = confusion_matrix(y_val, y_val_pred)
plt.figure()
sns.heatmap(val_conf_mat, annot=True)
plt.title("2A - Validation Confusion Matrix (MLFFNN)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/2A_MLFFNN_val_confmat.png")
plt.show()

y_test_pred = best_model.predict(X_test)
print("Test Accuracy:", 100*np.sum(y_test_pred==y_test)/y_test.size)
test_conf_mat = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(test_conf_mat, annot=True)
plt.title("2A - Test Confusion Matrix (MLFFNN)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/2A_MLFFNN_test_confmat.png")
plt.show()


# In[ ]:




