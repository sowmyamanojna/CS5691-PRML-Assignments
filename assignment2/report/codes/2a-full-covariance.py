#!/usr/bin/env python
# coding: utf-8

#########################################################################
import time
import numpy as np
import pandas as pd
from gmm import GMM
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import defaultdict
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split

plt.rcParams["font.size"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = 8,6
plt.rcParams['font.serif'] = "Cambria"
plt.rcParams['font.family'] = "serif"

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


#########################################################################
df = pd.read_csv("../datasets/2A/consolidated_train.csv")
X = df.drop("class", axis=1).to_numpy()
df.head()


#########################################################################
classes = np.unique(df["class"])
gmm_list = defaultdict(list)
q_list = list(range(2,23))  

for i in classes:
    print("="*50)
    df_select = df[df["class"]==i]
    X_select = df_select.drop("class", axis=1).to_numpy()
    for q in q_list:
        gmm = GMM(q=q)
        gmm.fit(X_select)
        gmm_list[i].append(gmm)


#########################################################################
import pickle
fin = open("2a_gmm_results", "wb")
pickle.dump(gmm_list, fin)
fin.close()


#########################################################################
df_test = pd.read_csv("../datasets/2A/consolidated_dev.csv")
df_cv = df_test.sample(frac=0.7)
X_cv = df_cv.drop("class", axis=1).to_numpy()
display(df_cv.head())
df_test = df_test.drop(df_cv.index)
X_test = df_test.drop("class", axis=1).to_numpy()
display(df_test.head())


#########################################################################
accuracy_list = []
test_accuracy_list = []
for i in tqdm(range(len(q_list))):
    gmm0 = gmm_list[0][i]
    gmm1 = gmm_list[1][i]
    gmm2 = gmm_list[2][i]
    gmm3 = gmm_list[3][i]
    gmm4 = gmm_list[4][i]
    
    # Training
    a = gmm0.indv_log_likelihood(X)
    b = gmm1.indv_log_likelihood(X)
    c = gmm2.indv_log_likelihood(X)
    d = gmm3.indv_log_likelihood(X)
    e = gmm4.indv_log_likelihood(X)

    f = np.hstack((a, b, c, d, e))
    pred = np.argmax(f, axis=1)
    accuracy_list.append(np.sum(pred == df["class"])/df["class"].size)
    
    # Testing
    a = gmm0.indv_log_likelihood(X_test)
    b = gmm1.indv_log_likelihood(X_test)
    c = gmm2.indv_log_likelihood(X_test)
    d = gmm3.indv_log_likelihood(X_test)
    e = gmm4.indv_log_likelihood(X_test)

    f = np.hstack((a, b, c, d, e))
    pred = np.argmax(f, axis=1)
    test_accuracy_list.append(np.sum(pred == df_test["class"])/df_test["class"].size)


#########################################################################
plt.plot(q_list, accuracy_list, '.-')
plt.title("Accuracy across varying Q")
plt.xlabel("Q for each class")
plt.ylabel("Accuracy")
plt.show()

plt.plot(q_list, cv_accuracy_list, '.-')
plt.title("CV Accuracy across varying Q")
plt.xlabel("Q for each class")
plt.ylabel("Accuracy")
plt.show()

plt.plot(q_list, test_accuracy_list, '.-')
plt.title("Test Accuracy across varying Q")
plt.xlabel("Q for each class")
plt.ylabel("Accuracy")
plt.show()


#########################################################################
import seaborn as sns
from sklearn.metrics import confusion_matrix

best_model = np.argmax(acc["Sum"])

gmm0 = gmm_list[0][best_model]
gmm1 = gmm_list[1][best_model]
gmm2 = gmm_list[2][best_model]
gmm3 = gmm_list[3][best_model]
gmm4 = gmm_list[4][best_model]

# Training
a = gmm0.indv_log_likelihood(X)
b = gmm1.indv_log_likelihood(X)
c = gmm2.indv_log_likelihood(X)
d = gmm3.indv_log_likelihood(X)
e = gmm4.indv_log_likelihood(X)

f = np.hstack((a, b, c, d, e))
pred = np.argmax(f, axis=1)
conf_mat = confusion_matrix(pred, df["class"])
plt.figure()
sns.heatmap(conf_mat, annot=True)
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# CV
a = gmm0.indv_log_likelihood(X_cv)
b = gmm1.indv_log_likelihood(X_cv)
c = gmm2.indv_log_likelihood(X_cv)
d = gmm3.indv_log_likelihood(X_cv)
e = gmm4.indv_log_likelihood(X_cv)

f = np.hstack((a, b, c, d, e))
pred = np.argmax(f, axis=1)
conf_mat = confusion_matrix(pred, df_cv["class"])
plt.figure()
sns.heatmap(conf_mat, annot=True)
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# Testing
a_test = gmm0.indv_log_likelihood(X_test)
b_test = gmm1.indv_log_likelihood(X_test)
c_test = gmm2.indv_log_likelihood(X_test)
d_test = gmm3.indv_log_likelihood(X_test)
e_test = gmm4.indv_log_likelihood(X_test)

f_test = np.hstack((a_test, b_test, c_test, d_test, e_test))
pred_test = np.argmax(f_test, axis=1)
conf_mat = confusion_matrix(pred_test, df_test["class"])
plt.figure()
sns.heatmap(conf_mat, annot=True)
plt.title("Testing Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()


#########################################################################
import seaborn as sns
from sklearn.metrics import confusion_matrix

gmm0 = gmm_list[0][0]
gmm1 = gmm_list[1][0]
gmm2 = gmm_list[2][0]
gmm3 = gmm_list[3][4]
gmm4 = gmm_list[4][3]

# Training
a = gmm0.indv_log_likelihood(X)
b = gmm1.indv_log_likelihood(X)
c = gmm2.indv_log_likelihood(X)
d = gmm3.indv_log_likelihood(X)
e = gmm4.indv_log_likelihood(X)

f = np.hstack((a, b, c, d, e))
pred = np.argmax(f, axis=1)
conf_mat = confusion_matrix(pred, df["class"])
plt.figure()
sns.heatmap(conf_mat, annot=True)
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# CV
a = gmm0.indv_log_likelihood(X_cv)
b = gmm1.indv_log_likelihood(X_cv)
c = gmm2.indv_log_likelihood(X_cv)
d = gmm3.indv_log_likelihood(X_cv)
e = gmm4.indv_log_likelihood(X_cv)

f = np.hstack((a, b, c, d, e))
pred = np.argmax(f, axis=1)
conf_mat = confusion_matrix(pred, df_cv["class"])
plt.figure()
sns.heatmap(conf_mat, annot=True)
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# Testing
a_test = gmm0.indv_log_likelihood(X_test)
b_test = gmm1.indv_log_likelihood(X_test)
c_test = gmm2.indv_log_likelihood(X_test)
d_test = gmm3.indv_log_likelihood(X_test)
e_test = gmm4.indv_log_likelihood(X_test)

f_test = np.hstack((a_test, b_test, c_test, d_test, e_test))
pred_test = np.argmax(f_test, axis=1)
conf_mat = confusion_matrix(pred_test, df_test["class"])
plt.figure()
sns.heatmap(conf_mat, annot=True)
plt.title("Testing Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
