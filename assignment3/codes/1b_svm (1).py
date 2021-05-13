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
plt.rcParams["font.size"]=18
plt.rcParams["axes.grid"]=True
plt.rcParams["figure.figsize"]=12,8
plt.rcParams["font.serif"]="Cambria"
plt.rcParams["font.family"]="serif"
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import seaborn as sns
color_list=["springgreen","gold","palevioletred","cyan"]


# ## Importing and splitting the 1b datasets

# In[2]:


cols=["x1","x2","y"]
train_data=pd.read_csv("train1b.csv",names=cols)
dev_data=pd.read_csv("dev1b.csv",names=cols)
data_cv,data_test=train_test_split(dev_data,test_size=0.3,random_state=42)
X_train=train_data[["x1","x2"]].to_numpy()
y_train=train_data["y"].to_numpy().astype("int")

X_cv=data_cv[["x1","x2"]].to_numpy()
y_cv=data_cv["y"].to_numpy().astype("int")

X_test=data_test[["x1","x2"]].to_numpy()
y_test=data_test["y"].to_numpy().astype("int")


# In[3]:


plt.scatter(train_data["x1"],train_data["x2"],c=[color_list[i] for i in y_train])


# # Training the polynomial Kernel:

# In[4]:


C_list=[1,10,100,1000]
degree_list=[1,2,3,4,5,6]
gamma_list=[1,0.1,0.01,"auto"]
coef0_list=[10,100]


# In[5]:


param_grid={"C":C_list,"degree":degree_list,"gamma":gamma_list,"coef0":coef0_list,"kernel":["poly"]}


# In[6]:


grid=GridSearchCV(svm.SVC(),param_grid,verbose=7,return_train_score=True,cv=2)


# In[7]:


grid.fit(X_train,y_train)


# In[8]:


results_df=pd.DataFrame(grid.cv_results_)


# In[9]:


results_df=results_df.sort_values(by="rank_test_score")


# In[40]:


results_df.head(10)


# In[48]:


results_df["params"].iloc[9]


# In[12]:


print("Best Parameters Choosen:")
for i in grid.best_params_:
    print("   - ", i, ": ", grid.best_params_[i], sep="")


# In[13]:


best_poly=svm.SVC(C=1000,coef0=100,degree=5,gamma=0.1,kernel="poly")


# In[14]:


best_poly.fit(X_train,y_train)
y_cv_polypred=best_poly.predict(X_cv)
y_test_polypred=best_poly.predict(X_test)
y_train_polypred=best_poly.predict(X_train)


# In[15]:


y_poly_trainaccuracy=100*np.sum(y_train_polypred==y_train)/y_train.size
y_poly_cvaccuracy=100*np.sum(y_cv_polypred==y_cv)/y_cv.size


# In[16]:


y_poly_trainaccuracy


# In[17]:


y_poly_cvaccuracy


# In[18]:


y_poly_testaccuracy=100*np.sum(y_test_polypred==y_test)/y_test.size


# In[19]:


y_poly_testaccuracy


# In[20]:


conf_mat=confusion_matrix(y_train,y_train_polypred)
plt.figure()
sns.heatmap(conf_mat,annot=True)
plt.title("1B - Train Confusion Matrix (SVM with Polynomial Kernel)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1B_SVM_poly_train_confmat.png")
plt.show()

print(" Test Accuracy:",y_poly_testaccuracy)
test_conf_mat=confusion_matrix(y_test,y_test_polypred)
plt.figure()
sns.heatmap(test_conf_mat,annot=True)
plt.title("1B - Test Confusion Matrix (SVM with Polynomial Kernel)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1B_SVM_poly_Test_confmat.png")
plt.show()


# # Decision Region Plot for Polynomial Kernel:

# In[21]:


sv=(best_poly.support_vectors_)


# In[ ]:





# In[49]:


h=0.1
x1_min,x1_max=train_data["x1"].min()-1,train_data["x1"].max()+1
x2_min,x2_max=train_data["x2"].min()-1,train_data["x2"].max()+1
xx,yy=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
z=best_poly.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure()
plt.contour(xx,yy,z,np.unique(z).size-1,colors=color_list,alpha=1)
plt.contourf(xx,yy,z,np.unique(z).size-1,colors=color_list,alpha=0.1)
plt.scatter(train_data["x1"],train_data["x2"],c=[color_list[i] for i in y_train])
plt.scatter(sv[:,0],sv[:,1],marker="x",c="k",label="Support Vectors",alpha=0.5)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("1B - Decision Region Plot (Polynomial SVM)")
plt.savefig("images/1B_SVM_poly_decision_plot.png")
plt.show()


# In[ ]:





# # Training the Gaussian Kernel:

# In[23]:


gamma_list=[1,0.01,0.001,0.0001]
C_list=[0.1,1,10,100,1000]

gauss_cv_accuracy={}
gauss_train_accuracy={}
for i in gamma_list:
    gauss_train_accuracy[i]=[]
    gauss_cv_accuracy[i]=[]
    for j in C_list:
        model=svm.SVC(kernel="rbf",decision_function_shape="ovr",C=j,gamma=i)
        model.fit(X_train,y_train)
        ytrain_pred=model.predict(X_train)
        ycv_pred=model.predict(X_cv)
        gauss_train_accuracy[i].append(100*np.sum(ytrain_pred==y_train)/y_train.size)
        gauss_cv_accuracy[i].append(100*np.sum(ycv_pred==y_cv)/y_cv.size)


# In[24]:


gauss_cv_accuracy;


# In[25]:


gauss_accuracy_table=pd.DataFrame()


# In[26]:


gauss_accuracy_table["gamma"]=[1,1,1,1,1,0.01,0.01,0.01,0.01,0.01,0.001,0.001,0.001,0.001,0.001,0.0001,0.0001,0.0001,0.0001,0.0001]


# In[27]:


gauss_accuracy_table["C"]=[0.1,1,10,100,1000,0.1,1,10,100,1000,0.1,1,10,100,1000,0.1,1,10,100,1000]


# In[28]:


val_ac=[]
for i in (gamma_list):
    for j in range(5):
        val_ac.append(gauss_cv_accuracy[i][j])


# In[29]:


gauss_accuracy_table["validation_accuracy"]=val_ac


# In[30]:


train_ac=[]
for i in (gamma_list):
    for j in range(5):
        train_ac.append(gauss_train_accuracy[i][j])


# In[31]:


gauss_accuracy_table["Train accuracy"]=train_ac


# In[32]:


gauss_accuracy_table


# In[33]:


best_gauss_model=svm.SVC(kernel="rbf",decision_function_shape="ovr",C=1,gamma=1)
best_gauss_model.fit(X_train,y_train)
ygauss_testpred=best_gauss_model.predict(X_test)


# In[34]:


test_gauss_accuracy=100*np.sum(ygauss_testpred==y_test)/y_test.size


# In[35]:


test_gauss_accuracy


# # Confusion matrix for train and test data set, best gaussian model

# In[36]:


ytraingauss_pred=best_gauss_model.predict(X_train)
print(" Train Accuracy:",100*np.sum(ytraingauss_pred==y_train)/y_train.size)
conf_mat=confusion_matrix(y_train,ytraingauss_pred)
plt.figure()
sns.heatmap(conf_mat,annot=True)
plt.title("1B - Train Confusion Matrix (SVM with Gaussian Kernel)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1B_SVM_gauss_train_confmat.png")
plt.show()

print(" Test Accuracy:",test_gauss_accuracy)
test_conf_mat=confusion_matrix(y_test,ygauss_testpred)
plt.figure()
sns.heatmap(test_conf_mat,annot=True)
plt.title("1B - Test Confusion Matrix (SVM with Gaussian Kernel)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1B_SVM_gauss_Test_confmat.png")
plt.show()




# # Decision function plot:

# In[37]:


sv_gauss=best_gauss_model.support_vectors_


# In[38]:


h=0.1
x1_min,x1_max=train_data["x1"].min()-1,train_data["x1"].max()+1
x2_min,x2_max=train_data["x2"].min()-1,train_data["x2"].max()+1
xx,yy=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
z=best_gauss_model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.figure()
plt.contour(xx,yy,z,np.unique(z).size-1,colors=color_list,alpha=1)
plt.contourf(xx,yy,z,np.unique(z).size-1,colors=color_list,alpha=0.1)
plt.scatter(train_data["x1"],train_data["x2"],c=[color_list[i] for i in y_train])
plt.scatter(sv_gauss[:,0],sv_gauss[:,1],marker="x",c="k",label="Support Vectors",alpha=1)
plt.legend()
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("1B - Decision Region Plot (Gaussian SVM)")
plt.savefig("images/1B_SVM_gauss_decision_plot.png")
plt.show()


# In[ ]:




