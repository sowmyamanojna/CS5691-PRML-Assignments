#!/usr/bin/env python
# coding: utf-8
#########################################################################
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from gmm import GMM_vl

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#########################################################################
df = pd.read_csv("../datasets/2B/consolidated_train.csv")
X = df.drop(["class","image", "index"], axis=1).to_numpy()
print(X.shape)
df.head()

#########################################################################
classes = np.unique(df["class"])
gmm_list = []

for i in classes:
    gmm = GMM_vl(q=14)
    df_selected = df[df["class"]==i]

    X_selected = df_selected.drop(["class", "image", "index"], axis=1).to_numpy()
    gmm.fit(X_selected, epochs=20)
    gmm_list.append(gmm)

#########################################################################
gmm.probab(df_selected)

#########################################################################
gmm.gamma

#########################################################################


