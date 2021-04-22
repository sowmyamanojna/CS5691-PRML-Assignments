import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_consolidated_data2A(classes_present):
    df = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in classes_present:
        df_new = pd.read_csv("../datasets/2A/"+i+"/train.csv")
        # df_new = pd.read_csv("../datasets/2A/"+i+"/train.csv", nrows=182)
        df_new["image_names"] = classes_present[i]
        df_new = df_new.rename(columns={"image_names":"class"})
        df = df.append(df_new)

        df_new_test = pd.read_csv("../datasets/2A/"+i+"/dev.csv")
        # df_new_test = pd.read_csv("../datasets/2A/"+i+"/dev.csv", nrows=52)
        df_new_test["image_names"] = classes_present[i]
        df_new_test = df_new_test.rename(columns={"image_names":"class"})
        df_test = df_test.append(df_new_test)

    df.to_csv("../datasets/2A/consolidated_train.csv", index=False)
    df_test.to_csv("../datasets/2A/consolidated_dev.csv", index=False)
    # df.to_csv("../datasets/2A/consolidated_train_small.csv", index=False)
    # df_test.to_csv("../datasets/2A/consolidated_dev_small.csv", index=False)

def get_consolidated_data2B(classes_present):
    df = pd.DataFrame()
    df_test = pd.DataFrame()
    
    for i in classes_present:
        files = os.listdir("../datasets/2B/"+i+"/train/")
        for k,j in tqdm(enumerate(files)):
            df_new = pd.read_csv("../datasets/2B/"+i+"/train/"+j, header=None, sep=" ")
            df_new["class"] = classes_present[i]
            df_new = df_new.reset_index()
            df_new["image"] = str(k)
            df = df.append(df_new)

        files = os.listdir("../datasets/2B/"+i+"/dev/")
        for k,j in tqdm(enumerate(files)):
            df_new_test = pd.read_csv("../datasets/2B/"+i+"/dev/"+j, header=None, sep=" ")
            df_new_test["class"] = classes_present[i]
            df_new_test = df_new_test.reset_index()
            df_new_test["image"] = str(k)
            df_test = df.append(df_new_test)

    df.to_csv("../datasets/2B/consolidated_train.csv", index=False)
    df_test.to_csv("../datasets/2B/consolidated_dev.csv", index=False)


if __name__ == "__main__":
    classes_present = {"coast":0, "highway":1, "mountain":2, "opencountry":3, "tallbuilding":4}
    get_consolidated_data2B(classes_present)