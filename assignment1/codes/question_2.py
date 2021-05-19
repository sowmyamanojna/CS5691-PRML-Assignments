################################################################
from IPython import get_ipython

import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

################################################################
# # Task 2

################################################################
func2d=pd.read_csv("function1_2d.csv",index_col = 0)

################################################################
# ## 2.1 Generating the polynomial basis functions of degrees 2, 3 and 6: ( Sp to dataset 2)

################################################################
### Creating the polynomial basis functions of degree M and number of examples = n:

def create_phi(M,n,x1,x2):
    d=2
    D = int(ma.factorial(d+M)/(ma.factorial(d)*ma.factorial(M)))
    phi = np.zeros((n,D))
    
    if M == 2:
        exp_ar = [[0,0],[1,0],[0,1],[2,0],[0,2],[1,1]]
    if M == 3:
        exp_ar = [[0,0],[1,0],[0,1],[2,0],[0,2],[1,1],[2,1],[1,2],[3,0],[0,3]]
    if M == 6:
        exp_ar = [[0,0],[1,0],[0,1],[2,0],[0,2],[1,1],[2,1],[1,2],[3,0],\
        [0,3],[3,1],[1,3],[2,2],[4,0],[0,4],[4,1],[1,4],[2,3],[3,2],[5,0],\
        [0,5],[5,1],[1,5],[2,4],[4,2],[3,3],[6,0],[0,6]]
    for i in range(D):
        phi[:,i] = (x1**(exp_ar[i][0]))*(x2**(exp_ar[i][1]))
    return(phi)

################################################################
# ## 2.2 Solving for optimal parameters using regularization, lambda=0  for unregularized: (versatile) 
################################################################
# ### The function regularized_pseudo_inv(lamb,X) returns:
# $$(\lambda I+X^TX)^{-1}X^T$$ 
# 
# Where lambda is the hyperparameter in the quadratic regularization

################################################################
def regularized_pseudo_inv(lamb,phi):
    return(np.matmul(np.linalg.inv(lamb*np.identity(phi.shape[1])+np.matmul(np.transpose(phi),phi)),np.transpose(phi)))

################################################################
# ### The function opt_regularized_param(lamb,phi,y) returns an array of optimal parameter values calculated using regularized cost function for an input design matrix phi, hyperparameter lambda and output values y. 

################################################################
def opt_regularized_param(lamb,phi,y):
    return(np.matmul(regularized_pseudo_inv(lamb,phi),y))

################################################################
# ## 2.3 The function y_pred(X,w) returns predicted function values for input matrix X and set of chosen parameter values w a: (versatile)
# $$y=Xw$$

################################################################
def y_pred(phi,w):
    return(np.matmul(phi,w))

################################################################
# ## 2.4 Splitting the data into train, cross-validation and test: (versatile)

################################################################
def create_datasets(data,train_size,cv_size):
    data.sample(frac=1).reset_index(drop=True)
    data_train=data[0:train_size]
    data_cv=data[train_size:train_size+cv_size]
    data_test=data[cv_size+train_size:]
    return(data_train,data_cv,data_test)
    


################################################################
def split_cols(data_train,data_cv,data_test):
    #x1_train,x2_train,y_train,x1_cv,x2_cv,y_cv,x1_test,x2_test,y_test
    x1_train=np.array(data_train)[:,0]
    x2_train=np.array(data_train)[:,1]
    y_train=np.array(data_train)[:,2]
    x1_cv=np.array(data_cv)[:,0]
    x2_cv=np.array(data_cv)[:,1]
    y_cv=np.array(data_cv)[:,2]
    x1_test=np.array(data_test)[:,0]
    x2_test=np.array(data_test)[:,1]
    y_test=np.array(data_test)[:,2]
    
    return(x1_train,x2_train,y_train,x1_cv,x2_cv,y_cv,x1_test,x2_test,y_test)
    


################################################################
### Function to calculate RMSE:

def RMSE(y_pred, t):
    n = len(y_pred)
    return(np.sqrt(np.sum((y_pred - t)**2)/n))

################################################################
# ## 2.5 Predicting for degree 2, train size 50:

################################################################
data_train,data_cv,data_test=create_datasets(func2d,50,30)


################################################################
x1_train,x2_train,y_train,x1_cv,x2_cv,y_cv,x1_test,x2_test,y_test=split_cols(data_train,data_cv,data_test)


################################################################
### design matrix:
phi_train=create_phi(2,len(y_train),x1_train,x2_train)
phi_cv=create_phi(2,len(y_cv),x1_cv,x2_cv)
phi_test=create_phi(2,len(y_test),x1_test,x2_test)


################################################################
data_train.head()


################################################################
y_testpred50={}
y_trainpred50={}
y_cvpred50={}
lambda_list=[0,0.5,1,2,10,50,100]
rmse_train50=[]
rmse_test50=[]
rmse_cv50=[]


################################################################
for l in lambda_list:
    w=opt_regularized_param(l,phi_train,y_train);
    y_trainpred50[l]=y_pred(phi_train,w)
    y_testpred50[l]=y_pred(phi_test,w);
    y_cvpred50[l]=y_pred(phi_cv,w);
    rmse_train50.append(RMSE(y_trainpred50[l],y_train))
    rmse_test50.append(RMSE(y_testpred50[l],y_test))
    rmse_cv50.append(RMSE(y_cvpred50[l],y_cv))
    
    


################################################################
data50=pd.DataFrame(list(zip(lambda_list,rmse_train50,rmse_cv50,rmse_test50)),columns=["Lambda", "RMSE Train","RMSE CV","RMSE test"])


################################################################
data50

################################################################
# ## 2.6 Predicting for degree of complexity = 2 and train data size = 100

################################################################
data_train,data_cv,data_test=create_datasets(func2d,100,30)


################################################################
x1_train,x2_train,y_train,x1_cv,x2_cv,y_cv,x1_test,x2_test,y_test=split_cols(data_train,data_cv,data_test)


################################################################
### design matrix:
phi_train=create_phi(2,len(y_train),x1_train,x2_train)
phi_cv=create_phi(2,len(y_cv),x1_cv,x2_cv)
phi_test=create_phi(2,len(y_test),x1_test,x2_test)


################################################################
y_testpred100={}
y_trainpred100={}
y_cvpred100={}
lambda_list=[0,0.5,1,2,10,50,100]
rmse_train100=[]
rmse_test100=[]
rmse_cv100=[]


