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

# In[10]:


C_list=[0.1,1,10,100,1000]
cv_accuracy=[]
train_accuracy=[]
for i in C_list:
    model=svm.SVC(kernel="linear",decision_function_shape="ovo",C=i)
    model.fit(X_train,y_train)
    ytrain_pred=model.predict(X_train)
    ycv_pred=model.predict(X_cv)
    train_accuracy.append(100*np.sum(ytrain_pred==y_train)/y_train.size)
    cv_accuracy.append(100*np.sum(ycv_pred==y_cv)/y_cv.size)


# In[11]:


accuracy_table=pd.DataFrame()
accuracy_table["C"]=C_list
accuracy_table["training accuracy"]=train_accuracy
accuracy_table["validation accuracy"]=cv_accuracy
accuracy_table


# ## we proceed with C=0.1: 

# In[14]:


best_svc=svm.SVC(kernel="linear",C=0.1,decision_function_shape="ovo")
best_svc.fit(X_train,y_train)
ytrain_pred=best_svc.predict(X_train)
print(" Train Accuracy:",100*np.sum(ytrain_pred==y_train)/y_train.size)
conf_mat=confusion_matrix(y_train,ytrain_pred)
plt.figure()
sns.heatmap(conf_mat,annot=True)
plt.title("1A - Train Confusion Matrix (Linear SVM)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1A_SVM_train_confmat.png")
plt.show()

ycv_pred=best_svc.predict(X_cv)
print(" Validation Accuracy:",100*np.sum(ycv_pred==y_cv)/y_cv.size)
val_conf_mat=confusion_matrix(y_cv,ycv_pred)
plt.figure()
sns.heatmap(val_conf_mat,annot=True)
plt.title("1A - Validation Confusion Matrix (Linear SVM)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1A_SVM_val_confmat.png")
plt.show()

ytest_pred=best_svc.predict(X_test)
print(" Test Accuracy:",100*np.sum(ytest_pred==y_test)/y_test.size)
test_conf_mat=confusion_matrix(y_test,ytest_pred)
plt.figure()
sns.heatmap(test_conf_mat,annot=True)
plt.title("1A - Test Confusion Matrix (Linear SVM)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.savefig("images/1A_SVM_Test_confmat.png")
plt.show()



# ## Visiualizing decision boundaries:

# In[15]:


h=0.1
x1_min,x1_max=train_data["x1"].min()-1,train_data["x1"].max()+1
x2_min,x2_max=train_data["x2"].min()-1,train_data["x2"].max()+1
xx,yy=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))


# In[16]:


z=best_svc.predict(np.c_[xx.ravel(),yy.ravel()])
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


# ## Linear SVM classifier for every pair of classes:

# In[17]:


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
    plt.plot(x2,yx)
    
    yx=a*x2-(predictor.intercept_[0]-1)/w[1]
    plt.plot(x2,yx,"k--")
    
    yx=a*x2-(predictor.intercept_[0]+1)/w[1]
    plt.plot(x2,yx,"k--")
    c1=color_list[y1]
    c2=color_list[y2]
    colors_list=[c1,c2]

    plt.contourf(xx,yy,z,np.unique(z).size-1,colors=color,alpha=0.25)
    plt.scatter(df2["x1"],df2["x2"],c=[color_list[i] for i in df2["y"].astype(int)],label=(y1,y2))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
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
            


# In[18]:


linear_ovo_plot(1,2,train_data,"images/1A_ovo_12.png","Support vectors and Boundary region between y=1.0 and y=2.0",color=[color_list[1],color_list[2]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf12_train.png",conf_test_save_name="images/1A_ovo_conf12_test.png",df_dev=dev_data)


# In[19]:


linear_ovo_plot(1,3,train_data,"images/1A_ovo_13.png","Support vectors and Boundary region between y=1.0 and y=3.0",color=[color_list[1],color_list[3]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf13_train.png",conf_test_save_name="images/1A_ovo_conf13_test.png",df_dev=dev_data)


# In[20]:


linear_ovo_plot(0,1,train_data,"images/1A_ovo_01.png","Support vectors and Boundary region between y=0.0 and y=1.0",color=[color_list[0],color_list[1]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf01_train.png",conf_test_save_name="images/1A_ovo_conf01_test.png",df_dev=dev_data)


# In[21]:


linear_ovo_plot(0,2,train_data,"images/1A_ovo_02.png","Support vectors and Boundary region between y=0.0 and y=2.0",color=[color_list[0],color_list[2]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf02_train.png",conf_test_save_name="images/1A_ovo_conf02_test.png",df_dev=dev_data)


# In[22]:


linear_ovo_plot(0,3,train_data,"images/1A_ovo_03.png","Support vectors and Boundary region between y=0.0 and y=3.0",color=[color_list[0],color_list[3]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf03_train.png",conf_test_save_name="images/1A_ovo_conf03_test.png",df_dev=dev_data)


# In[23]:


linear_ovo_plot(2,3,train_data,"images/1A_ovo_23.png","Support vectors and Boundary region between y=2.0 and y=3.0",color=[color_list[2],color_list[3]],conf_title_train="Confusion Matrix on train data",conf_title_test="Confusion matrix on test data",conf_train_save_name="images/1A_ovo_conf23_train.png",conf_test_save_name="images/1A_ovo_conf23_test.png",df_dev=dev_data)


# In[ ]:




