#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.metrics import confusion_matrix


# In[3]:


from sklearn.metrics import classification_report


# #                                         Dataset 1a:

# ### Importing the train, test and cross validation data sets

# In[4]:


col_names=["x1","x2","y"]


# In[5]:


## Train data
data1a=pd.read_csv("train.csv",names=col_names)


# In[6]:


data1a.head()


# In[7]:


data1a.isnull().sum()


# In[8]:


data1a.describe()


# In[9]:


## Splitting the columns of train data

X1train=data1a["x1"]
X2train=data1a["x2"]
Ytrain=np.array(data1a["y"])
Xtrain=np.array(data1a.drop("y",axis=1))


# In[10]:


## group labels 
data1a["y"].unique()


# In[11]:


## Importing the test and cross-validation data
data1a_dev=pd.read_csv("dev.csv",names=col_names)


# In[12]:


## Function to split a given dataset into test and cross-validation

def create_datasets(data,cv_size):
    data.sample(frac=1).reset_index(drop=True)
    data_cv=data[0:cv_size]
    data_test=data[cv_size:]
    return(data_cv,data_test)


# In[13]:


def euclidean(p1,p2):
    d=np.linalg.norm(np.array(p1)-np.array(p2))
    return d


# In[14]:


def accuracy(y_pred,y_actual):
    true_count=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_actual[i]:
            true_count+=1;
    return(true_count/len(y_pred))


# In[15]:


data1a_dev.shape


# In[16]:


## Splitting in the ratio 70:30 (cv:test)
data1a_cv,data1a_test=create_datasets(data1a_dev,84)


# ### Plotting the train data set

# In[17]:


X_cv=np.array(data1a_cv.drop("y",axis=1))
Y_cv=np.array(data1a_cv["y"])
X_test=np.array(data1a_test.drop("y",axis=1))
Y_test=np.array(data1a_test["y"])

plt.figure()
plt.scatter(X1train[Ytrain==0],X2train[Ytrain==0],label="y=0")
plt.scatter(X1train[Ytrain==1],X2train[Ytrain==1],label="y=1")
plt.scatter(X1train[Ytrain==2],X2train[Ytrain==2],label="y=2")
plt.scatter(X1train[Ytrain==3],X2train[Ytrain==3],label="y=3")
plt.legend()
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Scatter plot of data 1a")
plt.savefig("Scatter plot of data_1a.jpg")
plt.show()


# # K Nearest Neighbour Classifier for dataset 1a:

# In[18]:


def knn(x,y,test,k):
    distances=[]
    for i in range(len(x)):
        d=euclidean(x[i],test)
        l=(d,x[i],y[i])
        distances.append(l)
    distances.sort(key = lambda x:x[0])
    count=Counter()
    for i in distances[:k]:
        count[i[2]]+=1
    pred=count.most_common(1)[0][0]
    return(distances[:k],pred)
    


# ### KNN on given cross-validation and test datasets:

# In[19]:


k_list=[1,7,15]
Accuracy_cv=[]
Accuracy_train=[]
Accuracy_test=[]


# In[20]:


## iterating over k-values
for i in k_list:
    ycv_pred=[]
    for j in X_cv:
        ycv_pred.append(knn(Xtrain,Ytrain,j,i)[1])
    ytest_pred=[]
    for j in X_test:
        ytest_pred.append(knn(Xtrain,Ytrain,j,i)[1])
    ytrain_pred=[]
    for j in Xtrain:
        ytrain_pred.append(knn(Xtrain,Ytrain,j,i)[1])
    Accuracy_cv.append(accuracy(Y_cv,ycv_pred))
    Accuracy_test.append(accuracy(Y_test,ytest_pred))
    Accuracy_train.append(accuracy(Ytrain,ytrain_pred))


# In[21]:


accuracy_table_knn=pd.DataFrame(list(zip(k_list,Accuracy_train,Accuracy_cv,Accuracy_test)),columns=["k-value", "Accuracy train","Accuracy CV","Accuracy test"])


# In[22]:


accuracy_table_knn


# In[23]:


cm=confusion_matrix(Ytrain,ytrain_pred,labels=[1.0,3.0,0.0,2.0])
cm2=confusion_matrix(Y_test,ytest_pred)


# In[24]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[25]:


cmd=ConfusionMatrixDisplay(cm,display_labels=[0.0,1.0,2.0,3.0])
plt.figure()
cmd.plot()
plt.savefig("1a_cm_knn_train.jpg")


# In[26]:


cmd2=ConfusionMatrixDisplay(cm2,display_labels=[0.0,1.0,2.0,3.0])
plt.figure()
cmd2.plot()
plt.savefig("1a_cm_knn_test.jpg")


# # Naive Bayes Classifier:

# In[27]:


def seperate_by_classval(data):
    ## the target variable must be stored in a column named "y"
    class_vals=list(data["y"].unique())
    seperated=dict()
    features=data.drop('y',axis=1)
    Y=np.array(data["y"])
    ## creates a key value corresponding to each class label
    for i in class_vals:
        seperated[i]=features[Y==i];
    return(seperated)


# In[28]:


def priori(data):
    seperated_data=seperate_by_classval(data)
    probs=dict()
    for i in seperated_data.keys():
        probs[i]=len(seperated_data[i])/len(data);
    return probs


# In[29]:


def mu_sigma(data):
    seperated_data=seperate_by_classval(data)
    mean=dict()
    sigma={}
    for i in list(seperated_data.keys()):
        features=seperated_data[i]
        mean[i]=[]
        sigma[i]=[]
        for j in range(seperated_data[i].shape[1]):
            mean[i].append(np.mean(features.iloc[:,j]))
            sigma[i].append(np.std(features.iloc[:,j]))
    return(mean,sigma)  


# In[30]:


def gauss_val(x,cov_matrix,mean):
    x=np.array(x)
    A=(x-mean)
    B=np.linalg.inv(cov_matrix)
    C=np.transpose(A)
    det=np.linalg.det(cov_matrix)
    AB=A.dot(B)
    m=AB.dot(C)
    
    exp_term=np.exp(-m/2)
    d=2
    return (exp_term/(2*np.pi*det**0.5))


# ## Seperating the data according to class label:

# In[31]:


seperated_data=seperate_by_classval(data1a)


# In[32]:


### Labels: 
labels=list(data1a["y"].unique())


# In[33]:


### Initiating the accuracy table
accuracy_table_bayes=pd.DataFrame()
accuracy_table_bayes["method"]=["Ci=Cj=sigma**2*I","Ci=Cj=C","Ci!=Cj"]


# In[34]:


accuracy_table_bayes["Train Accuracy"]=[0,0,0]
accuracy_table_bayes["CV accuracy"]=[0,0,0]
accuracy_table_bayes["Test Accuracy"]=[0,0,0]


# ### Case 1: Ci=Cj=sigma**2 * I

# In[35]:


sigma=mu_sigma(data1a)[1]


# In[36]:


sigma


# In[37]:


var=0
for i in labels:
    var+=sigma[i][0]**2+sigma[i][1]**2
    
var=var/(4*2)    


# In[38]:


def predictor1(x):
    pyi_x={}
    pyi=priori(data1a)
    means=mu_sigma(data1a)[0]
    for i in labels:
        pyi_x[i]=pyi[i]*gauss_val(x,var*np.eye(2),means[i])
    val=sum(pyi_x.values())
    p=0
    for i in labels:
        pyi_x[i]/=val
        if pyi_x[i]>p:
            prediction=i
            p=pyi_x[i]
        
    
    return(pyi_x,prediction)


# In[39]:


predictor1([-10,5])


# In[40]:


Y_nb1_cv=[]
Y_nb1_test=[]
Y_nb1_train=[]
for i in range(len(X_cv)):
    Y_nb1_cv.append(predictor1(X_cv[i])[1])
for i in range(len(X_test)):
    Y_nb1_test.append(predictor1(X_test[i])[1])
for i in range(len(Xtrain)):
    Y_nb1_train.append(predictor1(Xtrain[i])[1])
    


# In[41]:


accuracy_table_bayes.iloc[0,1:]=[accuracy(Y_nb1_train,Ytrain),accuracy(Y_nb1_cv,Y_cv),accuracy(Y_nb1_test,Y_test)]


# ### Confusion Matrix

# In[42]:


cm_nb_train=confusion_matrix(Y_nb1_train,Ytrain)
cm_nb_test=confusion_matrix(Y_nb1_test,Y_test)


# In[43]:


len(Y_nb1_train)


# In[44]:


cmd_nb_train=ConfusionMatrixDisplay(cm_nb_train,display_labels=[0.0,1.0,2.0,3.0])
plt.figure()
cmd_nb_train.plot()
plt.savefig("1a_cm_nb_train.jpg")


# In[45]:


cmd_nb_test=ConfusionMatrixDisplay(cm_nb_test,display_labels=[0.0,1.0,2.0,3.0])
plt.figure()
cmd_nb_test.plot()
plt.savefig("1a_cm_nb_test.jpg")


# ## level curves:

# In[74]:


x, y = np.mgrid[-13:13:30j, -13:13:30j]
xy = np.column_stack([x.flat, y.flat])
#var = 0.6815614964194181
mu=mu_sigma(data1a)
z0 = np.zeros(len(xy))
z1 = np.zeros(len(xy))
z2 = np.zeros(len(xy))
z3 = np.zeros(len(xy))

for i in range(len(xy)):

    z0[i] = gauss_val(xy[i],var*np.eye(2),mu[0][0])
    z1[i] = gauss_val(xy[i],var*np.eye(2),mu[0][1])
    z2[i] = gauss_val(xy[i],var*np.eye(2),mu[0][2])
    z3[i] = gauss_val(xy[i],var*np.eye(2),mu[0][3])


z0 = z0.reshape(x.shape)


z1 = z1.reshape(x.shape)

z2 = z2.reshape(x.shape)
z3 = z3.reshape(x.shape)
color_list = ["springgreen", "mediumturquoise", "palevioletred","red"]
plt.figure()
data1a.plot.scatter("x1", "x2", c=[color_list[int(i)] for i in data1a["y"]], alpha=1)
#plt.contourf(x, y, classes, 2, colors=color_list, alpha=0.1)
#plt.contour(x, y, classes, 2, colors=color_list, alpha=1)
plt.contour(x, y, z0, levels=np.logspace(-5,5,20), colors=color_list[0])
plt.contour(x, y, z1, levels=np.logspace(-5,5,20), colors=color_list[1])
plt.contour(x, y, z2, levels=np.logspace(-5,5,20), colors=color_list[2])
plt.contour(x, y, z3, levels=np.logspace(-5,5,20), colors=color_list[3])

plt.title("Decision Boundaries + Contours - Diagonal Covariance")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("contour1b_case1.png")
plt.show()


# In[ ]:





# ### Case 2: Covariance matrix is same for all the classes:

# In[46]:


cov_matrix={}
for i in labels:
    cov_matrix[i]=np.cov(seperated_data[i],rowvar=False)  


# In[47]:


cov_matrix


# In[48]:


C=np.zeros((2,2))
for i in labels:
    C+=cov_matrix[i]
C/=4


# In[49]:


C


# In[50]:


def predictor2(x):
    pyi_x={}
    pyi=priori(data1a)
    means=mu_sigma(data1a)[0]
    for i in labels:
        pyi_x[i]=pyi[i]*gauss_val(x,C,means[i])
    val=sum(pyi_x.values())
    p=0
    for i in labels:
        pyi_x[i]/=val
        if pyi_x[i]>p:
            prediction=i
            p=pyi_x[i]
        
    
    return(pyi_x,prediction)


# In[51]:


Y_nb2_cv=[]
Y_nb2_test=[]
Y_nb2_train=[]
for i in range(len(X_cv)):
    Y_nb2_cv.append(predictor2(X_cv[i])[1])
for i in range(len(X_test)):
    Y_nb2_test.append(predictor2(X_test[i])[1])
for i in range(len(Xtrain)):
    Y_nb2_train.append(predictor2(Xtrain[i])[1])
    


# In[52]:


accuracy_table_bayes.iloc[1,1:]=[accuracy(Y_nb2_train,Ytrain),accuracy(Y_nb2_cv,Y_cv),accuracy(Y_nb2_test,Y_test)]


# ## level curves

# In[78]:


x, y = np.mgrid[-13:13:30j, -13:13:30j]
xy = np.column_stack([x.flat, y.flat])
z0 = np.zeros(len(xy))
z1 = np.zeros(len(xy))
z2 = np.zeros(len(xy))
z3 = np.zeros(len(xy))

for i in range(len(xy)):

    z0[i] = gauss_val(xy[i],C,mu[0][0])
    z1[i] = gauss_val(xy[i],C,mu[0][1])
    z2[i] = gauss_val(xy[i],C,mu[0][2])
    z3[i] = gauss_val(xy[i],C,mu[0][3])


z0 = z0.reshape(x.shape)


z1 = z1.reshape(x.shape)

z2 = z2.reshape(x.shape)
z3 = z3.reshape(x.shape)
color_list = ["springgreen", "mediumturquoise", "palevioletred","red"]
plt.figure()
data1a.plot.scatter("x1", "x2", c=[color_list[int(i)] for i in data1a["y"]], alpha=1)
#plt.contourf(x, y, classes, 2, colors=color_list, alpha=0.1)
#plt.contour(x, y, classes, 2, colors=color_list, alpha=1)
plt.contour(x, y, z0, levels=np.logspace(-5,5,20), colors=color_list[0])
plt.contour(x, y, z1, levels=np.logspace(-5,5,20), colors=color_list[1])
plt.contour(x, y, z2, levels=np.logspace(-5,5,20), colors=color_list[2])
plt.contour(x, y, z3, levels=np.logspace(-5,5,20), colors=color_list[3])

plt.title("Contours - Case b")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("contour1b_case2.png")
plt.show()


# In[ ]:





# ###  Case 3: Covariance matrix is different for all the classes: 

# In[53]:


def predictor3(x):
    pyi_x={}
    pyi=priori(data1a)
    means=mu_sigma(data1a)[0]
    for i in labels:
        pyi_x[i]=pyi[i]*gauss_val(x,cov_matrix[i],means[i])
    val=sum(pyi_x.values())
    p=0
    for i in labels:
        pyi_x[i]/=val
        if pyi_x[i]>p:
            prediction=i
            p=pyi_x[i]
        
    
    return(pyi_x,prediction)
        


# In[54]:


predictor3([5,5])


# In[55]:


Y_nb3_cv=[]
Y_nb3_test=[]
Y_nb3_train=[]
for i in range(len(X_cv)):
    Y_nb3_cv.append(predictor3(X_cv[i])[1])
for i in range(len(X_test)):
    Y_nb3_test.append(predictor3(X_test[i])[1])
for i in range(len(Xtrain)):
    Y_nb3_train.append(predictor3(Xtrain[i])[1])
    


# In[56]:


accuracy_table_bayes.iloc[2,1:]=[accuracy(Y_nb3_train,Ytrain),accuracy(Y_nb3_cv,Y_cv),accuracy(Y_nb3_test,Y_test)]


# In[57]:


accuracy_table_bayes


# In[77]:


x, y = np.mgrid[-13:13:30j, -13:13:30j]
xy = np.column_stack([x.flat, y.flat])
z0 = np.zeros(len(xy))
z1 = np.zeros(len(xy))
z2 = np.zeros(len(xy))
z3 = np.zeros(len(xy))

for i in range(len(xy)):

    z0[i] = gauss_val(xy[i],cov_matrix[0],mu[0][0])
    z1[i] = gauss_val(xy[i],cov_matrix[1],mu[0][1])
    z2[i] = gauss_val(xy[i],cov_matrix[2],mu[0][2])
    z3[i] = gauss_val(xy[i],cov_matrix[3],mu[0][3])


z0 = z0.reshape(x.shape)


z1 = z1.reshape(x.shape)

z2 = z2.reshape(x.shape)
z3 = z3.reshape(x.shape)
color_list = ["springgreen", "mediumturquoise", "palevioletred","red"]
plt.figure()
data1a.plot.scatter("x1", "x2", c=[color_list[int(i)] for i in data1a["y"]], alpha=1)
#plt.contourf(x, y, classes, 2, colors=color_list, alpha=0.1)
#plt.contour(x, y, classes, 2, colors=color_list, alpha=1)
plt.contour(x, y, z0, levels=np.logspace(-5,5,20), colors=color_list[0])
plt.contour(x, y, z1, levels=np.logspace(-5,5,20), colors=color_list[1])
plt.contour(x, y, z2, levels=np.logspace(-5,5,20), colors=color_list[2])
plt.contour(x, y, z3, levels=np.logspace(-5,5,20), colors=color_list[3])

plt.title("Contours - Case c")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("contour1b_case3.png")
plt.show()


# ### Confusion matrix for naive bayes classifier:

# In[ ]:





# ### Decision boundary plot for knn:

# In[58]:


min1,max1=data1a["x1"].min()-1,data1a["x1"].max()+1
min2,max2=data1a["x2"].min()-1,data1a["x2"].max()+1


# In[59]:


resolution=0.5
x1grid=np.arange(min1,max1,resolution)
x2grid=np.arange(min2,max2,resolution)


# In[60]:


xx,yy=np.meshgrid(x1grid,x2grid)


# In[61]:


r1,r2=xx.flatten(),yy.flatten()
r1,r2=r1.reshape((len(r1),1)),r2.reshape((len(r2),1))


# In[62]:


grid=np.hstack((r1,r2))


# In[63]:


yhat_knn_1=[]
for i in range(len(grid)):
    yhat_knn_1.append(knn(Xtrain,Ytrain,grid[i,:],1)[1])


# In[64]:


len(grid)


# In[65]:


yhat_knn_1=np.array(yhat_knn_1)


# In[66]:


zz=yhat_knn_1.reshape(xx.shape)


# In[67]:


data1a["y"].unique()


# In[68]:


plt.figure()
plt.contourf(xx,yy,zz,alpha=0.5,cmap="Paired")
plt.scatter(X1train[Ytrain==0],X2train[Ytrain==0],label="y=0",c="Blue")
plt.scatter(X1train[Ytrain==1],X2train[Ytrain==1],label="y=1",c="Green")
plt.scatter(X1train[Ytrain==2],X2train[Ytrain==2],label="y=2",c="Orange")
plt.scatter(X1train[Ytrain==3],X2train[Ytrain==3],label="y=3",c='red')
plt.legend()
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision region plot of data 1a, knn classifier")
plt.savefig("1a_knn_decision_region.jpg")
plt.show()


# In[69]:


grid


# In[70]:


yhat_nb=[]
for i in range(len(grid)):
    yhat_nb.append(predictor1(grid[i,:])[1])


# In[71]:


yhat_nb=np.array(yhat_nb)


# In[72]:


zz_nb=yhat_nb.reshape(xx.shape)


# In[73]:


plt.figure()
plt.contourf(xx,yy,zz_nb,alpha=0.5,cmap="Paired")
plt.scatter(X1train[Ytrain==0],X2train[Ytrain==0],label="y=0",c="Blue")
plt.scatter(X1train[Ytrain==1],X2train[Ytrain==1],label="y=1",c="Green")
plt.scatter(X1train[Ytrain==2],X2train[Ytrain==2],label="y=2",c="Orange")
plt.scatter(X1train[Ytrain==3],X2train[Ytrain==3],label="y=3",c='red')
plt.legend()
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision region plot of data 1a,naive-bayes classifier")
plt.savefig("1a_nb_case1_decisionregion.jpg")
plt.show()


# In[ ]:




