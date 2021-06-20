import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_consolidated_data2B(classes_present, cwd):
    flag = 0
    os.chdir("../datasets/2B/")

    # Loop over each class
    for i in tqdm(classes_present):
        # Loop over each image data
        train_path = i+"/train/"
        dev_path = i+"/dev/"

        train_file_names = os.listdir(train_path)
        train_file_names = [i for i in train_file_names if "csv" not in i]

        dev_file_names = os.listdir(dev_path)
        dev_file_names = [i for i in dev_file_names if "csv" not in i]

        for j in train_file_names:
            file_path = train_path+j
            temp = pd.read_csv(file_path, header=None, sep=" ")
            temp = temp.to_numpy()

            for row in range(temp.shape[0]):
                name = i+"/train_"+str(row)+".csv"
                fout = open(name, "a")
                writer = csv.writer(fout)
                writer.writerow(temp[row,:])
                fout.close()

        for j in dev_file_names:
            file_path = dev_path+j
            temp = pd.read_csv(file_path, header=None, sep=" ")
            temp = temp.to_numpy()

            for row in range(temp.shape[0]):
                name = i+"/dev_"+str(row)+".csv"
                fout = open(name, "a")
                writer = csv.writer(fout)
                writer.writerow(temp[row,:])
                fout.close()


if __name__ == "__main__":
    classes_present = {"coast":0, "highway":1, "mountain":2, "opencountry":3, "tallbuilding":4}
    cwd = os.getcwd()

    get_consolidated_data2B(classes_present, cwd)