import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = 8,6
plt.rcParams['font.serif'] = "Cambria"
plt.rcParams['font.family'] = "serif"

def gaussian_basis_fit(num_clusters, df_new, df_val_new, df_test_new, df_train, df_val, df_test, lmbda=0, fit_var="T_min"):
    sse_list = []
    kmeans_list = []
    label_list = []
    val_label_list = []
    test_label_list = []
    cluster_centers_list = []
    error_list = []
    val_error_list = []
    test_error_list = []

    for n_clu in num_clusters:
        kmeans = KMeans(n_clusters=n_clu, random_state=42).fit(df_new.to_numpy())
        kmeans_list.append(kmeans)
        sse_list.append(kmeans.inertia_)
        label_list.append(kmeans.labels_)
        cluster_centers_list.append(kmeans.cluster_centers_)  

        mean_centers = cluster_centers_list[-1]
        corresponding_center = mean_centers[label_list[-1],:]

        X = df_new.to_numpy()
        X_val = df_val_new.to_numpy()
        X_test = df_test_new.to_numpy()

        val_labels = kmeans.predict(X_val)
        val_label_list.append(val_labels)

        test_labels = kmeans.predict(X_test)
        test_label_list.append(test_labels)

        distance = np.linalg.norm(X-corresponding_center, axis=1)
        var = np.var(distance)*distance.size

        phi = np.ones((X.shape[0], 1))
        phi_val = np.ones((X_val.shape[0], 1))
        phi_test = np.ones((X_test.shape[0], 1))

        for i in range(n_clu):
            A = X - mean_centers[i,:]
            A = np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var)
            
            B = X_val - mean_centers[i,:]
            B = np.exp(-np.linalg.norm(X_val-mean_centers[i,:], axis=1)**2/var)

            C = X_test - mean_centers[i,:]
            C = np.exp(-np.linalg.norm(X_test-mean_centers[i,:], axis=1)**2/var)
            
            phi = np.append(phi, np.exp(-np.linalg.norm(X-mean_centers[i,:], axis=1)**2/var).reshape(-1,1), axis=1)
            phi_val = np.append(phi_val, np.exp(-np.linalg.norm(X_val-mean_centers[i,:], axis=1)**2/var).reshape(-1,1), axis=1)
            phi_test = np.append(phi_test, np.exp(-np.linalg.norm(X_test-mean_centers[i,:], axis=1)**2/var).reshape(-1,1), axis=1)

        if fit_var == "T_min":
            W1 = (np.linalg.inv(phi.T @ phi + lmbda*np.identity(phi.shape[1])) @ phi.T) @ df_train["Next_Tmin"]
            W1 = W1.reshape(-1,1)
            
            pred = phi @ W1
            val_pred = phi_val @ W1
            test_pred = phi_test @ W1

            plt.figure(figsize=[15,6])
            plt.subplot(1, 2, 1)
            plt.hist(pred, alpha=0.5)
            plt.hist(df_train["Next_Tmin"], alpha=0.5)
            plt.title("Next_Tmin; Clusters: "+str(n_clu) + ", $\lambda$: "+str(lmbda))
            plt.subplot(1, 2, 2)
            plt.plot(df_train["Next_Tmin"], pred, ".", label="Prediction")
            plt.plot(df_train["Next_Tmin"], df_train["Next_Tmin"], label="True Value")
            plt.title("Next_Tmin; Clusters: "+str(n_clu) + ", $\lambda$: "+str(lmbda))
            plt.legend()
            name = "images/t3_d3/T_min_nclu_"+str(n_clu)+"_lambda_"+str(lmbda)+".png"
            plt.savefig(name)
            plt.show()

            error = np.linalg.norm(df_train["Next_Tmin"].to_numpy().reshape(-1,1)-pred)/(pred.size)**0.5
            error_list.append(error)

            val_error = np.linalg.norm(df_val["Next_Tmin"].to_numpy().reshape(-1,1)-val_pred)/(val_pred.size)**0.5
            val_error_list.append(val_error)

            test_error = np.linalg.norm(df_test["Next_Tmin"].to_numpy().reshape(-1,1)-test_pred)/(test_pred.size)**0.5
            test_error_list.append(test_error)

        elif fit_var == "T_max":
            W1 = (np.linalg.inv(phi.T @ phi + lmbda*np.identity(phi.shape[1])) @ phi.T) @ df_train["Next_Tmax"]
            W1 = W1.reshape(-1,1)
            
            pred = phi @ W1
            val_pred = phi_val @ W1
            test_pred = phi_test @ W1

            plt.figure(figsize=[15,6])
            plt.title("Clusters: "+str(n_clu))
            plt.subplot(1, 2, 1)
            plt.hist(pred, alpha=0.5)
            plt.hist(df_train["Next_Tmax"], alpha=0.5)
            plt.title("Next_Tmax; Clusters: "+str(n_clu) + ", $\lambda$: "+str(lmbda))
            plt.subplot(1, 2, 2)
            plt.plot(df_train["Next_Tmax"], pred, ".", label="Prediction")
            plt.plot(df_train["Next_Tmax"], df_train["Next_Tmax"], label="True Value")
            plt.title("Next_Tmax; Clusters: "+str(n_clu) + ", $\lambda$: "+str(lmbda))
            plt.legend()
            
            name = "images/t3_d3/T_max_nclu_"+str(n_clu)+"_lambda_"+str(lmbda)+".png"
            plt.savefig(name)
            plt.show()

            error = np.linalg.norm(df_train["Next_Tmax"].to_numpy().reshape(-1,1)-pred)/(pred.size)**0.5
            error_list.append(error)

            val_error = np.linalg.norm(df_val["Next_Tmax"].to_numpy().reshape(-1,1)-val_pred)/(val_pred.size)**0.5
            val_error_list.append(val_error)

            test_error = np.linalg.norm(df_test["Next_Tmax"].to_numpy().reshape(-1,1)-test_pred)/(test_pred.size)**0.5
            test_error_list.append(test_error)


    return sse_list, error_list, val_error_list, kmeans_list, label_list, cluster_centers_list, val_label_list, test_error_list, test_label_list