###########################################################################
from IPython import get_ipython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


###########################################################################
dataset2 = pd.read_csv('function1_2d.csv',index_col = 0)


###########################################################################
dataset2['y']


###########################################################################
num_clusters = [1]
num_clusters.extend(range(2,100,10))
#num_clusters.extend(range(15, 31, 5))
#num_clusters.extend(range(40, 101, 10))

sse_list = []
label_list = []
cluster_centers_list = []
error_list = []

for n_clu in num_clusters:
    kmeans = KMeans(n_clusters=n_clu, random_state=42).fit(dataset2.to_numpy())
    sse_list.append(kmeans.inertia_)
    label_list.append(kmeans.labels_)
    cluster_centers_list.append(kmeans.cluster_centers_)  

    mean_centers = cluster_centers_list[-1]
  # print("Mean shape:", mean_centers.shape)
    corresponding_center = mean_centers[label_list[-1],:]

    X = dataset2.to_numpy()
    distance = np.linalg.norm(X-corresponding_center, axis=1)
    var = np.var(distance)*distance.size

    phi = np.ones((X.shape[0], 1))
    for i in range(n_clu):
        A = X-mean_centers[i,:]
    # print("A shape:", A.shape)
        A = np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var)
    # print("A shape:", A.shape)
        phi = np.append(phi, np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var).reshape(-1,1), axis=1)

    lmbda = 0
    W1 = (np.linalg.inv(phi.T @ phi + lmbda*np.identity(phi.shape[1])) @ phi.T) @ dataset2["y"]
    W1 = W1.reshape(-1,1)
    pred = phi @ W1

    plt.figure(figsize=[12,8])
    plt.title("Clusters: "+str(n_clu))
    plt.subplot(1, 2, 1)
    plt.hist(pred, alpha=0.5)
    plt.hist(dataset2["y"], alpha=0.5)
    plt.title("Clusters: "+str(n_clu))
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(dataset2["y"], pred, ".")
    plt.plot(dataset2["y"], dataset2["y"], '.')
    plt.title("Clusters: "+str(n_clu))
    plt.grid()
    plt.savefig('')
    plt.show()
    error = np.linalg.norm(dataset2["y"].to_numpy().reshape(-1,1)-pred)
    error_list.append(error)


###########################################################################
plt.figure(figsize=[12,8])
plt.subplot(1,2,1)
plt.plot(num_clusters, sse_list)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Knee Plot for determining the number of clusters")
plt.grid()
plt.subplot(1,2,2)
plt.plot(num_clusters, error_list)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title(("L2 Error for fit"))
plt.grid()
plt.show()


###########################################################################
error_list = np.array(error_list)
df_error = pd.DataFrame({"Clusters":num_clusters, "Error":error_list})
df_error.sort_values(by=["Error"], ascending=True, inplace=True)
df_error


###########################################################################
def create_datasets(data,train_size,cv_size):
    data.sample(frac=1).reset_index(drop=True)
    data_train=data[0:train_size]
    data_cv=data[train_size:train_size+cv_size]
    data_test=data[cv_size+train_size:]
    return(data_train,data_cv,data_test)
    


###########################################################################
lambda_list = [0.01,0.1,1,5,10,50,100]
trDS2, cvDS2, tDS2 = create_datasets(dataset2,1400,400)


###########################################################################
def f(ds,l,n_clu):
    sse_list = []
    label_list = []
    cluster_centers_list = []

    kmeans = KMeans(n_clusters=n_clu, random_state=42).fit(ds.to_numpy())
    sse_list.append(kmeans.inertia_)
    label_list.append(kmeans.labels_)
    cluster_centers_list.append(kmeans.cluster_centers_)  

    mean_centers = cluster_centers_list[-1]
  # print("Mean shape:", mean_centers.shape)
    corresponding_center = mean_centers[label_list[-1],:]

    X = ds.to_numpy()
    distance = np.linalg.norm(X-corresponding_center, axis=1)
    var = np.var(distance)*distance.size

    phi = np.ones((X.shape[0], 1))
    for i in range(n_clu):
        A = X-mean_centers[i,:]
    # print("A shape:", A.shape)
        A = np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var)
    # print("A shape:", A.shape)
        phi = np.append(phi, np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var).reshape(-1,1), axis=1)
    #lmbda = 1
    return(phi,mean_centers,var)

########################################################################### [markdown]
#     W1 = (np.linalg.inv(phi.T @ phi + l*np.identity(phi.shape[1])) @ phi.T) @ dataset2["y"]
#     W1 = W1.reshape(-1,1)
#     pred = phi @ W1
#     return(W1,)
#     error = np.linalg.norm(dataset2["y"].to_numpy().reshape(-1,1)-pred)
#     error_list.append(error)

###########################################################################
optClds2 = 10


###########################################################################
error_tr = []
error_cv = []
error_t = []
for l in range(len(lambda_list)):
    phi_tr = f(trDS2,l,optClds2)[0]
    phi_cv = f(cvDS2,l,optClds2)[0]
    phi_t = f(tDS2,l,optClds2)[0]
    w = (np.linalg.inv(phi_tr.T @ phi_tr + l*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS2["y"]
    pred_cv = phi_cv @ w
    pred_t = phi_t @ w
    pred_tr = phi_tr @ w
    error_cv.append(np.linalg.norm(cvDS2["y"].to_numpy().reshape(-1,1)-pred_cv))
    error_t.append(np.linalg.norm(tDS2["y"].to_numpy().reshape(-1,1)-pred_t))
    error_tr.append(np.linalg.norm(trDS2["y"].to_numpy().reshape(-1,1)-pred_tr))


###########################################################################
pd.DataFrame(list(zip(lambda_list,error_tr,error_cv,error_t)),columns=["Lambda", "RMSE Train","RMSE CV","RMSE test"])


###########################################################################
plt.plot(lambda_list,error_cv)


###########################################################################
phi_tr = f(trDS2,l,optClds2)[0]
phi_t = f(tDS2,l,optClds2)[0]
w = (np.linalg.inv(phi_tr.T @ phi_tr + 0.01*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS2["y"]
pred_t = phi_t @ w
pred_tr = phi_tr @ w


###########################################################################
plt.scatter(trDS2.iloc[:,2],pred_tr)
plt.xlabel("Target output,training data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model output for linear regression in gaussian basis and quadratic regularization")
plt.savefig("scatter_ds2quad.png")
plt.show()


###########################################################################
plt.scatter(cvDS2.iloc[:,2],pred_cv)


###########################################################################
plt.scatter(tDS2.iloc[:,2],pred_t)
plt.xlabel("Target output,test data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model output for linear regression in gaussian basis and quadratic regularization")
plt.savefig("scatter_ds2quadtest.png")
plt.show()


###########################################################################



###########################################################################
def tikhanov_reg(phi,mu,sigma,l):
    K = len(mu)
    phiT = np.zeros((K+1,K+1))
    phiT[0,0] = 1
    for i in range(1,K+1):
        for j in range(1,K+1):
            phiT[i,j] = np.exp(-(np.linalg.norm(mu[i-1]-mu[j-1])**2)/sigma**2)
    #l = 300
    #print(phiT.shape)
    pinv = np.linalg.inv(phi.T @ phi+l*phiT) @ phi.T
    return(pinv)


###########################################################################
lambda_list = [0,0.01,0.1,1,5]#,10,50,100]


###########################################################################
error_tr = []
error_cv = []
error_t = []
for l in range(len(lambda_list)):
#for l in [1]:
    phi_tr,mu_list,sig = f(trDS2,l,optClds2)
    phi_cv = f(cvDS2,l,optClds2)[0]
    phi_t = f(tDS2,l,optClds2)[0]
    #print(phi_tr.shape)
    #print(mu_list.shape)
    tikh = tikhanov_reg(phi_tr,mu_list,sig,l)
    #print("tikh shape ", tikh.shape)
    #print("phi_tr shape ", phi_tr.shape)
    #print("trDS2_y shape ", trDS2["y"].shape)

    w = tikh @ trDS2["y"]
    pred_cv = phi_cv @ w
    pred_t = phi_t @ w
    pred_tr = phi_tr @ w
    error_cv.append(np.linalg.norm(cvDS2["y"].to_numpy().reshape(-1,1)-pred_cv))
    error_t.append(np.linalg.norm(tDS2["y"].to_numpy().reshape(-1,1)-pred_t))
    error_tr.append(np.linalg.norm(trDS2["y"].to_numpy().reshape(-1,1)-pred_tr))


###########################################################################
pd.DataFrame(list(zip(lambda_list,error_tr,error_cv,error_t)),columns=["Lambda", "RMSE Train","RMSE CV","RMSE test"])


###########################################################################
plt.plot(lambda_list,error_cv)
plt.title('Erms on cross-validation data vs regularization factor')
plt.xlabel('lambda')
plt.ylabel('rmse error')


###########################################################################
plt.hist(pred_tr, alpha=0.5)
plt.hist(trDS2["y"], alpha=0.5)


###########################################################################
error_list = np.array(error_list)
df_error = pd.DataFrame({"lambda":lambda_list, "Error":error_cv})
df_error.sort_values(by=["Error"], ascending=True, inplace=True)
df_error


###########################################################################
l = 0.01
phi_tr = f(trDS2,l,optClds2)[0]
phi_t = f(tDS2,l,optClds2)[0]
tikh = tikhanov_reg(phi_tr,mu_list,sig,0.01)
w = tikh @ trDS2["y"]
w = (np.linalg.inv(phi_tr.T @ phi_tr + 0.1*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS2["y"]
pred_t = phi_t @ w
pred_tr = phi_tr @ w


###########################################################################
plt.scatter(trDS2.iloc[:,2],pred_tr)
plt.xlabel("Target output,training data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model output for linear regression in gaussian basis and Tikhonov regularization")
plt.savefig("scatter_ds2tikhtr.png")
plt.show()


###########################################################################
plt.scatter(tDS2.iloc[:,2],pred_t)
plt.xlabel("Target output,test data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model output for linear regression in gaussian basis and Tikhonov regularization")
plt.savefig("scatter_ds2tikhtest.png")
plt.show()


###########################################################################



###########################################################################
df_new = pd.read_csv("processed_biasclean.csv",index_col = 0)


###########################################################################
df_new.head()


###########################################################################
trDS3, cvDS3, tDS3 = create_datasets(df_new,1400,400)


###########################################################################
num_clusters = [1]
num_clusters.extend(range(2,10))
num_clusters.extend(range(15, 31, 5))
num_clusters.extend(range(40, 101, 10))

sse_list = []
label_list = []
cluster_centers_list = []
error_list = []

for n_clu in num_clusters:
    kmeans = KMeans(n_clusters=n_clu, random_state=42).fit(df_new.to_numpy())
    sse_list.append(kmeans.inertia_)
    label_list.append(kmeans.labels_)
    cluster_centers_list.append(kmeans.cluster_centers_)  

    mean_centers = cluster_centers_list[-1]
  # print("Mean shape:", mean_centers.shape)
    corresponding_center = mean_centers[label_list[-1],:]

    X = df_new.to_numpy()
    distance = np.linalg.norm(X-corresponding_center, axis=1)
    var = np.var(distance)*distance.size

    phi = np.ones((X.shape[0], 1))
    for i in range(n_clu):
        A = X-mean_centers[i,:]
    # print("A shape:", A.shape)
        A = np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var)
    # print("A shape:", A.shape)
        phi = np.append(phi, np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var).reshape(-1,1), axis=1)

    lmbda = 0
    W1 = (np.linalg.inv(phi.T @ phi + lmbda*np.identity(phi.shape[1])) @ phi.T) @ df_new["Next_Tmin"]
    W1 = W1.reshape(-1,1)
    pred = phi @ W1

    plt.figure(figsize=[12,8])
    plt.title("Clusters: "+str(n_clu))
    plt.subplot(1, 2, 1)
    plt.hist(pred, alpha=0.5)
    plt.hist(df_new["Next_Tmin"], alpha=0.5)
    plt.title("Clusters: "+str(n_clu))
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(df_new["Next_Tmin"], pred, ".")
    plt.plot(df_new["Next_Tmin"], df_new["Next_Tmin"], '.')
    plt.title("Clusters: "+str(n_clu))
    plt.grid()
    plt.show()
    error = np.linalg.norm(df_new["Next_Tmin"].to_numpy().reshape(-1,1)-pred)
    error_list.append(error)


###########################################################################
plt.figure(figsize=[12,8])
plt.subplot(1,2,1)
plt.plot(num_clusters, sse_list)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Knee Plot for determining the number of clusters")
plt.grid()
plt.subplot(1,2,2)
plt.plot(num_clusters, error_list)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title(("L2 Error for fit"))
plt.grid()
plt.show()


###########################################################################
error_list = np.array(error_list)
df_error = pd.DataFrame({"Clusters":num_clusters, "Error":error_list})
df_error.sort_values(by=["Error"], ascending=True, inplace=True)
df_error


###########################################################################
optClds3 = 9


###########################################################################
lambda_list = [0.01,0.1,1,5,10]


###########################################################################
error_tr = []
error_cv = []
error_t = []
for l in range(len(lambda_list)):
    phi_tr = f(trDS3,l,optClds3)[0]
    phi_cv = f(cvDS3,l,optClds3)[0]
    phi_t = f(tDS3,l,optClds3)[0]
    w = (np.linalg.inv(phi_tr.T @ phi_tr + l*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS3['Next_Tmin']
    pred_cv = phi_cv @ w
    pred_t = phi_t @ w
    pred_tr = phi_tr @ w
    error_cv.append(np.linalg.norm(cvDS3['Next_Tmin'].to_numpy().reshape(-1,1)-pred_cv))
    error_t.append(np.linalg.norm(tDS3['Next_Tmin'].to_numpy().reshape(-1,1)-pred_t))
    error_tr.append(np.linalg.norm(trDS3['Next_Tmin'].to_numpy().reshape(-1,1)-pred_tr))


###########################################################################
pd.DataFrame(list(zip(lambda_list,error_tr,error_cv,error_t)),columns=["Lambda", "RMSE Train","RMSE CV","RMSE test"])


###########################################################################
plt.plot(lambda_list,error_cv)
plt.title('Error on cross-validation data vs regularization factor, Quadratic regularization')
plt.xlabel('lambda')
plt.ylabel('rmse error')


###########################################################################
error_list = np.array(error_list)
df_error = pd.DataFrame({"lambda":lambda_list, "Error":error_cv})
df_error.sort_values(by=["Error"], ascending=True, inplace=True)
df_error


###########################################################################
l = .1
phi_tr = f(trDS3,l,optClds3)[0]
phi_t = f(tDS3,l,optClds3)[0]
w = (np.linalg.inv(phi_tr.T @ phi_tr + l*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS3['Next_Tmin']
pred_t = phi_t @ w
pred_tr = phi_tr @ w


###########################################################################
plt.scatter(trDS3.iloc[:,17],pred_tr)
plt.xlabel("Target output,train data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model with quadratic regularization, for Next_Tmin")
plt.savefig("scatter_ds3quadtrainT_min.png")
plt.show()


###########################################################################
plt.scatter(tDS3.iloc[:,17],pred_t)
plt.xlabel("Target output,test data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model output with quadratic regularization, for Next_Tmin")
plt.savefig("scatter_ds3quadtestT_min.png")
plt.show()


###########################################################################
error_tr = []
error_cv = []
error_t = []
for l in range(len(lambda_list)):
    phi_tr = f(trDS3,l,optClds3)[0]
    phi_cv = f(cvDS3,l,optClds3)[0]
    phi_t = f(tDS3,l,optClds3)[0]
    w = (np.linalg.inv(phi_tr.T @ phi_tr + l*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS3['Next_Tmax']
    pred_cv = phi_cv @ w
    pred_t = phi_t @ w
    pred_tr = phi_tr @ w
    error_cv.append(np.linalg.norm(cvDS3['Next_Tmax'].to_numpy().reshape(-1,1)-pred_cv))
    error_t.append(np.linalg.norm(tDS3['Next_Tmax'].to_numpy().reshape(-1,1)-pred_t))
    error_tr.append(np.linalg.norm(trDS3['Next_Tmax'].to_numpy().reshape(-1,1)-pred_tr))


###########################################################################
pd.DataFrame(list(zip(lambda_list,error_tr,error_cv,error_t)),columns=["Lambda", "RMSE Train","RMSE CV","RMSE test"])


###########################################################################
plt.plot(lambda_list,error_cv)
plt.title('Error on cross-validation data vs regularization factor, Quadratic regularization')
plt.xlabel('lambda')
plt.ylabel('rmse error')


###########################################################################
error_list = np.array(error_list)
df_error = pd.DataFrame({"lambda":lambda_list, "Error":error_cv})
df_error.sort_values(by=["Error"], ascending=True, inplace=True)
df_error


###########################################################################
l = 1
phi_tr = f(trDS3,l,optClds3)[0]
phi_t = f(tDS3,l,optClds3)[0]
w = (np.linalg.inv(phi_tr.T @ phi_tr + l*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS3['Next_Tmax']
pred_t = phi_t @ w
pred_tr = phi_tr @ w


###########################################################################
plt.scatter(trDS3.iloc[:,18],pred_tr)
plt.xlabel("Target output,train data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model with quadratic regularization, for Next_Tmax")
plt.savefig("scatter_ds3quadtrainT_max.png")
plt.show()


###########################################################################
plt.scatter(tDS3.iloc[:,18],pred_t)
plt.xlabel("Target output,test data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model with quadratic regularization, for Next_Tmax")
plt.savefig("scatter_ds3quadtestT_max.png")
plt.show()


###########################################################################
#Tikhonov reg for "Next_Tmin"

error_tr = []
error_cv = []
error_t = []
for l in range(len(lambda_list)):
#for l in [1]:
    phi_tr,mu_list,sig = f(trDS3,l,optClds3)
    phi_cv = f(cvDS2,l,optClds3)[0]
    phi_t = f(tDS2,l,optClds3)[0]
    #print(phi_tr.shape)
    #print(mu_list.shape)
    tikh = tikhanov_reg(phi_tr,mu_list,sig,l)
    #print("tikh shape ", tikh.shape)
    #print("phi_tr shape ", phi_tr.shape)
    #print("trDS2_y shape ", trDS2["y"].shape)

    w = tikh @ trDS3['Next_Tmin']
    pred_cv = phi_cv @ w
    pred_t = phi_t @ w
    pred_tr = phi_tr @ w
    error_cv.append(np.linalg.norm(cvDS3['Next_Tmin'].to_numpy().reshape(-1,1)-pred_cv))
    error_t.append(np.linalg.norm(tDS3['Next_Tmin'].to_numpy().reshape(-1,1)-pred_t))
    error_tr.append(np.linalg.norm(trDS3['Next_Tmin'].to_numpy().reshape(-1,1)-pred_tr))


###########################################################################
pd.DataFrame(list(zip(lambda_list,error_tr,error_cv,error_t)),columns=["Lambda", "RMSE Train","RMSE CV","RMSE test"])


###########################################################################
plt.plot(lambda_list,error_cv)
plt.title('Error on cross-validation data vs regularization factor, tikhonov regularization for Dataset 3')
plt.xlabel('lambda')
plt.ylabel('rmse error')


###########################################################################
plt.hist(pred_tr, alpha=0.5)
plt.hist(trDS3["Next_Tmin"], alpha=0.5)


###########################################################################
error_list = np.array(error_list)
df_error = pd.DataFrame({"lambda":lambda_list, "Error":error_cv})
df_error.sort_values(by=["Error"], ascending=True, inplace=True)
df_error


###########################################################################
l = .1
phi_tr = f(trDS3,l,optClds3)[0]
phi_t = f(tDS3,l,optClds3)[0]
tikh = tikhanov_reg(phi_tr,mu_list,sig,0.01)
w = tikh @ trDS3["Next_Tmin"]
w = (np.linalg.inv(phi_tr.T @ phi_tr + 0.1*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS3["Next_Tmin"]
pred_t = phi_t @ w
pred_tr = phi_tr @ w


###########################################################################
plt.scatter(trDS3.iloc[:,17],pred_tr)
plt.xlabel("Target output,train data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model with Tikhonov regularization, for Next_Tmin")
plt.savefig("scatter_ds3tikhtrainT_min.png")
plt.show()


###########################################################################
plt.scatter(tDS3.iloc[:,17],pred_t)
plt.xlabel("Target output,test data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model with Tikhonov regularization, for Next_Tmin")
plt.savefig("scatter_ds3tikhtrainT_min.png")
plt.show()


###########################################################################
#Tikhonov reg for "Next_Tmax"
error_tr = []
error_cv = []
error_t = []
for l in range(len(lambda_list)):
#for l in [1]:
    phi_tr,mu_list,sig = f(trDS3,l,optClds3)
    phi_cv = f(cvDS2,l,optClds3)[0]
    phi_t = f(tDS2,l,optClds3)[0]
    #print(phi_tr.shape)
    #print(mu_list.shape)
    n = len(trDS3)
    tikh = tikhanov_reg(phi_tr,mu_list,sig,l)
    #print("tikh shape ", tikh.shape)
    #print("phi_tr shape ", phi_tr.shape)
    #print("trDS2_y shape ", trDS2["y"].shape)

    w = tikh @ trDS3['Next_Tmax']
    pred_cv = phi_cv @ w
    pred_t = phi_t @ w
    pred_tr = phi_tr @ w
    error_cv.append(np.linalg.norm(cvDS3['Next_Tmax'].to_numpy().reshape(-1,1)-pred_cv))
    error_t.append(np.linalg.norm(tDS3['Next_Tmax'].to_numpy().reshape(-1,1)-pred_t))
    error_tr.append(np.linalg.norm(trDS3['Next_Tmax'].to_numpy().reshape(-1,1)-pred_tr))


###########################################################################
pd.DataFrame(list(zip(lambda_list,error_tr,error_cv,error_t)),columns=["Lambda", "RMSE Train","RMSE CV","RMSE test"])


###########################################################################
plt.plot(lambda_list,error_cv)
plt.title('Error on cross-validation data vs regularization factor, tikhonov regularization for Dataset 3, for T_max')
plt.xlabel('lambda')
plt.ylabel('rmse error')


###########################################################################
plt.hist(pred_tr, alpha=0.5)
plt.hist(trDS3["Next_Tmax"], alpha=0.5)


###########################################################################
error_list = np.array(error_list)
df_error = pd.DataFrame({"lambda":lambda_list, "Error":error_cv})
df_error.sort_values(by=["Error"], ascending=True, inplace=True)
df_error


###########################################################################
l = .01
phi_tr = f(trDS3,l,optClds3)[0]
phi_t = f(tDS3,l,optClds3)[0]
tikh = tikhanov_reg(phi_tr,mu_list,sig,0.01)
w = tikh @ trDS3["Next_Tmax"]
w = (np.linalg.inv(phi_tr.T @ phi_tr + 0.1*np.identity(phi_tr.shape[1])) @ phi_tr.T) @ trDS3["Next_Tmax"]
pred_t = phi_t @ w
pred_tr = phi_tr @ w


###########################################################################
plt.scatter(trDS3.iloc[:,18],pred_tr)
plt.xlabel("Target output,train data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model output with Tikhonov regularization, for Next_Tmax")
plt.savefig("scatter_ds3tikhtrainT_max.png")
plt.show()


###########################################################################
plt.scatter(trDS3.iloc[:,18],pred_tr)
plt.xlabel("Target output,test data")
plt.ylabel("Model output")
plt.title("Scatter plot of target vs model with Tikhonov regularization, for Next_Tmax")
plt.savefig("scatter_ds3tikhtestT_min.png")
plt.show()


###########################################################################



