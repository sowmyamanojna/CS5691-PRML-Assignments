import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.size"] = 18
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.figsize"] = 12,8
plt.rcParams['font.serif'] = "Cambria"
plt.rcParams['font.family'] = "serif"


class Perceptron():
    def __init__(self,data,epochs = 25,learning_rate = 0.01,verbose = False):
        self.d = data.shape[1]-1
        self.x = np.array(data.iloc[:,:self.d])
        self.y = np.array(data.iloc[:,self.d])
        self.epoch = epochs
        self.len_data = data.shape[0]
        self.eta = learning_rate
        self.verbose = verbose
    def one_epoch(self,w):
        flag = 0
        lst = list(range(self.len_data))
        random.shuffle(lst)
        for j in range(self.len_data):
            m = lst.pop()
            if w@self.x[m]<0:
                s = -1
            else:
                s = 1
            delta = (self.y[m] - s)
            if delta==0:
                flag+=1
            w += (self.eta)*(delta/2)*self.x[m]
        return(w,flag)
    def train(self):
        w = np.ones(self.d)
        for i in range(self.epoch):
            result = self.one_epoch(w)
            w = result[0]
            if (self.verbose==True):
                print("No. of correctly classified data points in "+str(i)+"th epoch : ", result[1])
            if result[1] == self.len_data:
                break
        if (self.verbose==True):
            print("Convergence reached in "+str(i)+" epochs")
        self.W = w
        #return(self.W)
    def Y(self,m,c,xRange):
        return(m*xRange+c)
    
    def plot_decision_region(self, name, savefig = False):
        x1Max = max(np.array(self.x[:,1]))+0.5
        x1Min = min(np.array(self.x[:,1]))-0.5
        x2Max = max(np.array(self.x[:,2]))+0.5
        x2Min = min(np.array(self.x[:,2]))-0.5
        color_list = ["palevioletred","royalblue"]
        xx, yy = np.meshgrid(np.arange(x1Min, x1Max, .02), np.arange(x2Min, x2Max, .02))
        z = self.predict(np.c_[np.ones(xx.ravel().shape),xx.ravel(),yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, colors= color_list, alpha=0.1)
        plt.contour(xx, yy, z, colors=color_list, alpha=.1)
        plt.scatter(self.x[:,1],self.x[:,2], c=[color_list[j] for j in self.y==1.])
        #plt.plot(np.linspace(x1Min,x1Max),self.Y(-self.W[1]/self.W[2],-self.W[0]/self.W[2],np.linspace(x1Min,x1Max)),label = "Decision Region")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Decision region plot for the " + name + " data points")
        #plt.legend()
        if savefig == True:
            plt.savefig(name+"dec_reg_perceptron.png")
        plt.show()
        
    def predict(self,mat):
        return(np.sign(mat @ self.W))
    def accuracy(self,test_data):
        prediction = self.predict(test_data.iloc[:,:self.d])
        compare = prediction == test_data.iloc[:,self.d]
        return(np.sum(compare)/len(test_data))
    
    def confusionMatrix(self, dat, name, save_fig = False):
        prediction = self.predict(dat.iloc[:,:self.d])
        conf_mat = confusion_matrix(dat.iloc[:,self.d],prediction)
        plt.figure()
        sns.heatmap(conf_mat, annot=True)
        plt.title("1A - Confusion Matrix for " + name + " data (Perceptron)")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        if (save_fig == True):
            plt.savefig("perceptron_" + name + "_confmat.png")
        plt.show()
