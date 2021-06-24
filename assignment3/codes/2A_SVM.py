#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["font.size"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = 12,8
plt.rcParams["font.serif"] = "Cambria"
plt.rcParams["font.family"] = "serif"
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import seaborn as sns
color_list = ["springgreen","gold","palevioletred","cyan"]


# In[2]:


train_data = pd.read_csv("train_new.csv")
dev_data = pd.read_csv("dev_new.csv")
data_cv,data_test = train_test_split(dev_data,test_size=0.3,random_state=42)
X_train = train_data.drop("class",axis=1).to_numpy()
y_train = train_data["class"].to_numpy().astype("int")

X_cv = data_cv.drop("class",axis=1).to_numpy()
y_cv = data_cv["class"].to_numpy().astype("int")

X_test = data_test.drop("class",axis=1).to_numpy()
y_test = data_test["class"].to_numpy().astype("int")


# In[3]:


train_data.describe()


# In[4]:


dev_data.describe()


# In[5]:


plt.figure(figsize=(30,30))
cor = train_data.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
plt.show()


# In[ ]:


gamma_list = [50,1,0.01,0.001,0.0001,10,100,"auto","scale"]
C_list = [0.01,0.1,1,10,100,1000]

cv_accuracy = {}
train_accuracy = {}
for i in gamma_list:
    train_accuracy[i]=[]
    cv_accuracy[i]=[]
    for j in C_list:
        model = svm.SVC(kernel="rbf",decision_function_shape="ovr",C=j,gamma=i,probability=True)
        model.fit(X_train,y_train)
        ytrain_pred = model.predict(X_train)
        ycv_pred = model.predict(X_cv)
        train_accuracy[i].append(100*np.sum(ytrain_pred==y_train)/y_train.size)
        cv_accuracy[i].append(100*np.sum(ycv_pred==y_cv)/y_cv.size)


# In[ ]:


cv_accuracy


# In[22]:


C_list = [0.1,0.01,1,10,100,1000]
gamma_list = [0.1,0.01,1,5,10,100,1000,"auto","scale"]
param_grid = {"C":C_list,"gamma":gamma_list,"kernel":["rbf"],"tol":[0.1,0.01,1],"class_weight":["balanced",None],"break_ties":[True,False],"shrinking":[True,False]}
grid = GridSearchCV(svm.SVC(),param_grid,verbose=7,return_train_score=True,cv=2)


# In[23]:


grid.fit(X_train,y_train)


# In[67]:


results_df = pd.DataFrame(grid.cv_results_)


# In[73]:


results_df = results_df.sort_values(by="rank_test_score")
results_df = results_df.reset_index(drop=True)


# In[74]:


results_df.head(10)


# In[28]:


results_df.iloc[0,:]


# In[33]:


params = grid.best_params_


# In[34]:


params


# In[39]:


model = svm.SVC(C=10,break_ties=False,class_weight=None,gamma=1,kernel="rbf",shrinking=True,tol=0.01)


# In[40]:


model.fit(X_train,y_train)


# In[42]:


ytrain_pred = model.predict(X_train)
ytest_pred = model.predict(X_test)
ycv_pred = model.predict(X_cv)


# In[43]:


y_trainaccuracy = 100*np.sum(ytrain_pred==y_train)/y_train.size
y_cvaccuracy = 100*np.sum(ycv_pred==y_cv)/y_cv.size
y_testaccuracy = 100*np.sum(ytest_pred==y_test)/y_test.size


# In[44]:


y_trainaccuracy


# In[45]:


y_cvaccuracy


# In[46]:


y_testaccuracy


# In[48]:


conf_mat = confusion_matrix(y_train,ytrain_pred)
plt.figure()
sns.heatmap(conf_mat,annot=True)
plt.title("2A - Train Confusion Matrix (SVM with Gaussian Kernel)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/2A_SVM_gauss_train_confmat.png")
plt.show()

print(" Test Accuracy:",y_testaccuracy)
test_conf_mat = confusion_matrix(y_test,ytest_pred)
plt.figure()
sns.heatmap(test_conf_mat,annot=True)
plt.title("2A - Test Confusion Matrix (SVM with Gaussian Kernel)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/2A_SVM_gauss_Test_confmat.png")
plt.show()


# In[ ]:





# In[32]:


X_train


# In[49]:


results_df.iloc[0,:]


# In[122]:


results_df["params"][620]


# In[123]:


results_df["mean_train_score"][620]


# In[124]:


results_df["mean_test_score"][620]


# In[71]:


results_df.head()


# In[ ]:




