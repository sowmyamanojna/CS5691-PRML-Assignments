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


# In[31]:


from statistics import mode


# In[2]:


from sklearn.metrics import classification_report


# In[3]:


from sklearn.model_selection import GridSearchCV


# In[4]:


import seaborn as sns


# In[5]:


color_list=["springgreen","gold","palevioletred","royalblue"]


# In[6]:


cols=["x1","x2","y"]
train_data=pd.read_csv("train.csv",names=cols)
dev_data=pd.read_csv("dev.csv",names=cols)


# In[7]:


data_cv,data_test=train_test_split(dev_data,test_size=0.3,random_state=42)


# In[8]:


X_train=train_data[["x1","x2"]].to_numpy()
y_train=train_data["y"].to_numpy().astype("int")

X_cv=data_cv[["x1","x2"]].to_numpy()
y_cv=data_cv["y"].to_numpy().astype("int")

X_test=data_test[["x1","x2"]].to_numpy()
y_test=data_test["y"].to_numpy().astype("int")


# In[9]:


train_data.head()


# ## Training the Model 

# ## we proceed with C=1: 

# ## Linear SVM classifier for every pair of classes:

# In[15]:


def linear_ovo_plot(y1,y2,df,save_name,title,color,conf_title_train,conf_title_test,conf_train_save_name,conf_test_save_name,df_dev):
    df2=df.loc[df["y"].isin([y1,y2])]
    df2_dev=df_dev.loc[df_dev["y"].isin([y1,y2])]
    df2_cv,df2_test=train_test_split(df2_dev,test_size=0.3,random_state=42)
    predictor=svm.SVC(kernel="linear",C=1,decision_function_shape="ovo").fit(df2.iloc[:,:-1],df2.iloc[:,-1])
    h=0.1
    x1_min,x1_max=df2["x1"].min()-1,df2["x1"].max()+1
    x2_min,x2_max=df2["x2"].min()-1,df2["x2"].max()+1
    xx,yy=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
    z=predictor.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    
    w=predictor.coef_[0]
    a=-w[0]/w[1]
    

    plt.figure()
    x2=np.linspace(xx.min(),xx.max())
    yx=a*x2-predictor.intercept_[0]/w[1]
    plt.plot(x2,yx,label="Decision Boundary")
    
    yx=a*x2-(predictor.intercept_[0]-1)/w[1]
    plt.plot(x2,yx,"k--", label="Support Vector")
    
    yx=a*x2-(predictor.intercept_[0]+1)/w[1]
    plt.plot(x2,yx,"k--",label="Support Vector")
    c1=color_list[y1]
    c2=color_list[y2]
    colors_list=[c1,c2]

    plt.contourf(xx,yy,z,np.unique(z).size-1,colors=color,alpha=0.25)
    plt.scatter(df2["x1"],df2["x2"],c=[color_list[i] for i in df2["y"].astype(int)])
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.legend(loc="upper right")
    plt.savefig(save_name)
    plt.title(title)
    plt.show()
    
    y_train=df2["y"]
    ytrain_pred=predictor.predict(df2.iloc[:,:-1])
    
    y_cv=df2_cv["y"]
    y_test=df2_test["y"]
    ytest_pred=predictor.predict(df2_test.iloc[:,:-1])
    
    
    conf_mat=confusion_matrix(y_train,ytrain_pred)
    plt.figure()
    sns.heatmap(conf_mat,annot=True)
    plt.title(conf_title_train)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.savefig(conf_train_save_name)
    plt.show()
    
    conf_mat=confusion_matrix(y_test,ytest_pred)
    plt.figure()
    sns.heatmap(conf_mat,annot=True)
    plt.title(conf_title_test)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.savefig(conf_test_save_name)
    plt.show()
            


# In[16]:


linear_ovo_plot(1,2,train_data,"images/1A_ovo_12.png","Support vectors and Boundary region between y=1.0 and y=2.0",color=[color_list[1],color_list[2]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf12_train.png",conf_test_save_name="images/1A_ovo_conf12_test.png",df_dev=dev_data)


# In[17]:


linear_ovo_plot(1,3,train_data,"images/1A_ovo_13.png","Support vectors and Boundary region between y=1.0 and y=3.0",color=[color_list[1],color_list[3]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf13_train.png",conf_test_save_name="images/1A_ovo_conf13_test.png",df_dev=dev_data)


# In[18]:


linear_ovo_plot(0,1,train_data,"images/1A_ovo_01.png","Support vectors and Boundary region between y=0.0 and y=1.0",color=[color_list[0],color_list[1]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf01_train.png",conf_test_save_name="images/1A_ovo_conf01_test.png",df_dev=dev_data)


# In[19]:


linear_ovo_plot(0,2,train_data,"images/1A_ovo_02.png","Support vectors and Boundary region between y=0.0 and y=2.0",color=[color_list[0],color_list[2]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf02_train.png",conf_test_save_name="images/1A_ovo_conf02_test.png",df_dev=dev_data)


# In[20]:


linear_ovo_plot(0,3,train_data,"images/1A_ovo_03.png","Support vectors and Boundary region between y=0.0 and y=3.0",color=[color_list[0],color_list[3]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf03_train.png",conf_test_save_name="images/1A_ovo_conf03_test.png",df_dev=dev_data)


# In[21]:


linear_ovo_plot(2,3,train_data,"images/1A_ovo_23.png","Support vectors and Boundary region between y=2.0 and y=3.0",color=[color_list[2],color_list[3]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf23_train.png",conf_test_save_name="images/1A_ovo_conf23_test.png",df_dev=dev_data)


# # Using one-vs-one models to predict for a test sample:

# In[28]:


def class_model(df,y1,y2,C=1):
    df2=df.loc[df["y"].isin([y1,y2])]
    predictor=svm.SVC(kernel="linear",C=C,decision_function_shape="ovo").fit(df2.iloc[:,:-1],df2.iloc[:,-1])
    return(predictor)


# In[29]:


model01=class_model(train_data,0,1)
model02=class_model(train_data,0,2)
model03=class_model(train_data,0,3)
model12=class_model(train_data,1,2)
model13=class_model(train_data,1,3)
model23=class_model(train_data,2,3)


# In[54]:


from collections import Counter


# In[57]:


def ovo_predictor(x):
    c=[]
    c.append(model01.predict(x)[0])
    c.append(model02.predict(x)[0])
    c.append(model03.predict(x)[0])
    c.append(model12.predict(x)[0])
    c.append(model13.predict(x)[0])
    c.append(model23.predict(x)[0])
    count=Counter(c)
    freq=0
    label=0
    for i in count.keys():
        if count[i]>freq:
            freq=count[i]
            label=i
    return label


# In[58]:


ytrain_pred=[]
ycv_pred=[]
ytest_pred=[]
for i in range(len(X_train)):
    x=X_train[i,:].reshape(1,-1)
    ytrain_pred.append(ovo_predictor(x))
for x in X_cv:
    x=x.reshape(1,-1)
    ycv_pred.append(ovo_predictor(x))
for x in X_test:
    x=x.reshape(1,-1)
    ytest_pred.append(ovo_predictor(x))


# In[59]:


def accuracy(actual,predicted):
    return 100*np.sum(predicted==actual)/actual.size


# In[60]:


accuracy(y_train,ytrain_pred)


# In[67]:


conf_mat=confusion_matrix(y_train,ytrain_pred)
plt.figure()
sns.heatmap(conf_mat,annot=True)
plt.title("1a - Confusion matrix for train data" )
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1a_confmatrix_train.png" )
plt.show()
    
conf_mat=confusion_matrix(y_test,ytest_pred)
plt.figure()
sns.heatmap(conf_mat,annot=True)
plt.title("1a - Confusion matrix for test data")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1a_confmatrix_test.png")
plt.show()


# In[64]:


h=0.1
x1_min,x1_max=train_data["x1"].min()-1,train_data["x1"].max()+1
x2_min,x2_max=train_data["x2"].min()-1,train_data["x2"].max()+1
xx,yy=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
X=np.c_[xx.ravel(),yy.ravel()]
z=[]
for i in X:
    x=i.reshape(1,-1)
    z.append(ovo_predictor(x))
z=np.array(z)
z=z.reshape(xx.shape)
plt.figure()
plt.contour(xx,yy,z,np.unique(z).size-1,colors=color_list,alpha=1)
plt.contourf(xx,yy,z,np.unique(z).size-1,colors=color_list,alpha=0.25)
plt.scatter(train_data["x1"],train_data["x2"],c=[color_list[i] for i in y_train])
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("1A-Full Decision Region Plot(SVM)")
plt.savefig("images/1A_SVM_full_decision_plot.png")
plt.show()


# In[ ]:




