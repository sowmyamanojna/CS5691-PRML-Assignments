## CS5691 PRML Assignment 1
**Team 1**  
**Team Members:**  
N Sowmya Manojna   BE17B007  
Thakkar Riya Anandbhai  PH17B010   
Chaithanya Krishna Moorthy  PH17B011   


```python
# Install required Packages
# Uncomment if you are running for the firts time
# !pip install -r requirements.txt
# try:
#     !mkdir images
# except:
#     pass
```


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('science')
plt.rcParams['font.size'] = 18
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = 12,9
```


```python
df = pd.read_csv("../datasets/function1.csv", index_col=0)
df.sort_values(by=["x"], inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1731</th>
      <td>-1.995370</td>
      <td>20.924170</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>-1.994300</td>
      <td>20.697622</td>
    </tr>
    <tr>
      <th>730</th>
      <td>-1.989581</td>
      <td>20.418372</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1.988462</td>
      <td>20.505452</td>
    </tr>
    <tr>
      <th>546</th>
      <td>-1.988102</td>
      <td>20.644535</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_polynomial_features(X, degree):    
    X_new = np.ones(X.shape)
    for i in range(1, degree+1):
        X_new = np.append(X_new, X**i, axis=1)
        
    return X_new

def get_weights(X, y, lmbda=0):
    d = X.shape[1]
    W = ((np.linalg.inv(X.T @ X + lmbda*np.identity(d))) @ X.T) @ y
    return W

def get_predictions(W, X):
    y = X @ W
    return y

def get_rmse(W, X, y):
    y_pred = get_predictions(W, X)
    rmse = np.linalg.norm(y_pred-y)/(y.size)**0.5
    return rmse

def get_plot(X, y, y_pred, X_sampled, y_sampled, title="", fname="tmp"):
    plt.figure()
    plt.plot(X, y, label="True Value")
    if y_sampled.size >= 100:
        plt.plot(X_sampled, y_sampled, 'r.', alpha=0.5, label="Sampled points")
    else:
        plt.plot(X_sampled, y_sampled, 'ro', alpha=0.75, label="Sampled points")
    plt.plot(X, y_pred, label="Predicted Value")
    if title:
        plt.title(title)
    plt.xlabel("X-values")
    plt.ylabel("Y-values")
    plt.legend()
    plt.savefig("images/"+fname)
    plt.show()
    
def get_result(df, sample_size, degrees_allowed):
    df_sample = df.sample(n=sample_size, random_state=42)
    X_sampled = df_sample["x"].to_numpy().reshape(-1,1)
    y_sampled = df_sample["y"].to_numpy().reshape(-1,1)
    
    for degree in degrees_allowed:
        X_new_sampled = get_polynomial_features(X_sampled, degree)
        X_transformed = get_polynomial_features(X, degree)
        
        for lmbda in lmbda_list:
            W = get_weights(X_new_sampled, y_sampled, lmbda=lmbda)
            y_pred = get_predictions(W, X_transformed)
            
            title = "Curve Fitting - Degree: "+str(degree) \
            +"; Sample Size: "+str(sample_size)+"; $\lambda$: " \
            +str(lmbda)
            if lmbda == 0.5:
                lmbda = "0_5"
            fname = "d_"+str(degree)+"_size_"+str(sample_size)+"_l_"+str(lmbda)
            get_plot(X, y, y_pred, X_sampled, y_sampled, title, fname)
```


```python
lmbda_list = [0, 0.5, 1, 2, 10, 50, 100]
degrees_allowed = [2, 3, 6, 9]
datasizes_considered = [10, 200]
complete_dataset_size = df.shape[0]

X = df["x"].to_numpy().reshape(-1,1)
y = df["y"].to_numpy().reshape(-1,1)

for sample_size in datasizes_considered:
    get_result(sample_size, degrees_allowed)
    
```


    
![png](output_5_0.png)
    



    
![png](output_5_1.png)
    



    
![png](output_5_2.png)
    



    
![png](output_5_3.png)
    



    
![png](output_5_4.png)
    



    
![png](output_5_5.png)
    



    
![png](output_5_6.png)
    



    
![png](output_5_7.png)
    



    
![png](output_5_8.png)
    



    
![png](output_5_9.png)
    



    
![png](output_5_10.png)
    



    
![png](output_5_11.png)
    



    
![png](output_5_12.png)
    



    
![png](output_5_13.png)
    



    
![png](output_5_14.png)
    



    
![png](output_5_15.png)
    



    
![png](output_5_16.png)
    



    
![png](output_5_17.png)
    



    
![png](output_5_18.png)
    



    
![png](output_5_19.png)
    



    
![png](output_5_20.png)
    



    
![png](output_5_21.png)
    



    
![png](output_5_22.png)
    



    
![png](output_5_23.png)
    



    
![png](output_5_24.png)
    



    
![png](output_5_25.png)
    



    
![png](output_5_26.png)
    



    
![png](output_5_27.png)
    



    
![png](output_5_28.png)
    



    
![png](output_5_29.png)
    



    
![png](output_5_30.png)
    



    
![png](output_5_31.png)
    



    
![png](output_5_32.png)
    



    
![png](output_5_33.png)
    



    
![png](output_5_34.png)
    



    
![png](output_5_35.png)
    



    
![png](output_5_36.png)
    



    
![png](output_5_37.png)
    



    
![png](output_5_38.png)
    



    
![png](output_5_39.png)
    



    
![png](output_5_40.png)
    



    
![png](output_5_41.png)
    



    
![png](output_5_42.png)
    



    
![png](output_5_43.png)
    



    
![png](output_5_44.png)
    



    
![png](output_5_45.png)
    



    
![png](output_5_46.png)
    



    
![png](output_5_47.png)
    



    
![png](output_5_48.png)
    



    
![png](output_5_49.png)
    



    
![png](output_5_50.png)
    



    
![png](output_5_51.png)
    



    
![png](output_5_52.png)
    



    
![png](output_5_53.png)
    



    
![png](output_5_54.png)
    



    
![png](output_5_55.png)
    



```python

```


```python

```
