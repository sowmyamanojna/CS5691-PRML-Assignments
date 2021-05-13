import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

class GridSearch1A():
    def __init__(self, model, parameters, verbose=0):
        self.model = model
        self.parameters = parameters
        self.verbose = verbose
        params_list = []
        self.params_keys = self.parameters.keys()

        for hls in parameters["hidden_layer_sizes"]:
            for act in parameters["activation"]:
                for s in parameters["solver"]:
                    for bs in parameters["batch_size"]:
                        for a in parameters["alpha"]:
                            for lr in parameters["learning_rate"]:
                                params_list.append({"hidden_layer_sizes":hls, \
                                                    "activation":act, \
                                                    "solver":s, \
                                                    "batch_size":bs, \
                                                    "alpha":a, \
                                                    "learning_rate":lr})
        self.params_list = params_list

    def fit(self, X_train, y_train, X_val, y_val):
        self.cv_results_ = pd.DataFrame(columns=self.params_keys)
        
        self.params_ = defaultdict(list)
        self.acc_list_ = []
        self.val_acc_list_ = []
        self.t_inv_list_ = []

        for params in tqdm(self.params_list):
            st = time()
            mlp = MLPClassifier(random_state=1, **params)

            mlp.fit(X_train, y_train)
            et = time()

            y_pred = mlp.predict(X_train)
            acc = 100*np.sum(y_pred==y_train)/y_train.size

            y_val_pred = mlp.predict(X_val)
            val_acc = 100*np.sum(y_val_pred==y_val)/y_val.size

            for i in params:
                self.params_[i].append(params[i])

            self.acc_list_.append(acc)
            self.val_acc_list_.append(val_acc)
            self.t_inv_list_.append(1/(et-st))

        for i in params:
            self.cv_results_[i] = self.params_[i]

        self.cv_results_["accuracy"] = self.acc_list_
        self.cv_results_["val_accuracy"] = self.val_acc_list_
        self.cv_results_["sum_accuracy"] = self.cv_results_["accuracy"] + self.cv_results_["val_accuracy"]
        self.cv_results_["t_inv"] = self.t_inv_list_
        self.cv_results_ = self.cv_results_.sort_values(by=["val_accuracy", "accuracy", "sum_accuracy", "t_inv"], ascending=False, ignore_index=True)

        self.best_params_ = self.cv_results_.iloc[0].to_dict()
        del self.best_params_["accuracy"]
        del self.best_params_["val_accuracy"]
        del self.best_params_["sum_accuracy"]
        del self.best_params_["t_inv"]
        

class GridSearch1B():
    def __init__(self, model, parameters, verbose=0):
        self.model = model
        self.parameters = parameters
        self.verbose = verbose
        params_list = []
        self.params_keys = self.parameters.keys()

        for hls in parameters["hidden_layer_sizes"]:
            for act in parameters["activation"]:
                for bs in parameters["batch_size"]:
                    for a in parameters["alpha"]:
                        for lr in parameters["learning_rate"]:
                            for es in parameters["early_stopping"]:
                                params_list.append({"hidden_layer_sizes":hls, \
                                                    "early_stopping":es, \
                                                    "learning_rate":lr, \
                                                    "activation":act, \
                                                    "batch_size":bs, \
                                                    "alpha":a})
        self.params_list = params_list

    def fit(self, X_train, y_train, X_val, y_val):
        self.cv_results_ = pd.DataFrame(columns=self.params_keys)
        
        self.params_ = defaultdict(list)
        self.acc_list_ = []
        self.val_acc_list_ = []
        self.t_inv_list_ = []

        for params in tqdm(self.params_list):
            st = time()
            mlp = MLPClassifier(random_state=1, **params)

            mlp.fit(X_train, y_train)
            et = time()

            y_pred = mlp.predict(X_train)
            acc = 100*np.sum(y_pred==y_train)/y_train.size

            y_val_pred = mlp.predict(X_val)
            val_acc = 100*np.sum(y_val_pred==y_val)/y_val.size

            for i in params:
                self.params_[i].append(params[i])

            self.acc_list_.append(acc)
            self.val_acc_list_.append(val_acc)
            self.t_inv_list_.append(1/(et-st))

        for i in params:
            self.cv_results_[i] = self.params_[i]

        self.cv_results_["accuracy"] = self.acc_list_
        self.cv_results_["val_accuracy"] = self.val_acc_list_
        self.cv_results_["sum_accuracy"] = self.cv_results_["accuracy"] + self.cv_results_["val_accuracy"]
        self.cv_results_["t_inv"] = self.t_inv_list_
        self.cv_results_ = self.cv_results_.sort_values(by=["val_accuracy", "accuracy", "sum_accuracy", "t_inv"], ascending=False, ignore_index=True)

        self.best_params_ = self.cv_results_.iloc[0].to_dict()
        self.best_params_["early_stopping"] = bool(self.best_params_["early_stopping"])
        del self.best_params_["accuracy"]
        del self.best_params_["val_accuracy"]
        del self.best_params_["sum_accuracy"]
        del self.best_params_["t_inv"]

class GridSearch2A():
    def __init__(self, model, parameters, verbose=0):
        self.model = model
        self.parameters = parameters
        self.verbose = verbose
        params_list = []

        for nc in parameters["pca__n_components"]:
            for hls in parameters["mlp__hidden_layer_sizes"]:
                for bs in parameters["mlp__batch_size"]:
                    for a in parameters["mlp__alpha"]:
                        for lr in parameters["mlp__learning_rate"]:
                            for es in parameters["mlp__early_stopping"]:
                                params_list.append([{"n_components":nc},
                                                {
                                                "hidden_layer_sizes":hls, \
                                                "early_stopping":es, \
                                                "learning_rate":lr, \
                                                "batch_size":bs, \
                                                "alpha":a, \
                                                }])

        self.params_list = params_list

    def fit(self, X_train, y_train, X_val, y_val):
        self.cv_results_ = pd.DataFrame()
        
        self.params_ = defaultdict(list)
        self.acc_list_ = []
        self.val_acc_list_ = []
        self.t_inv_list_ = []

        for params in tqdm(self.params_list):
            st = time()
            model = Pipeline([('pca', PCA(**params[0])), \
                              ('mlp', MLPClassifier(max_iter=500, random_state=1, **params[1]))])

            model.fit(X_train, y_train)
            et = time()

            y_pred = model.predict(X_train)
            acc = 100*np.sum(y_pred==y_train)/y_train.size

            y_val_pred = model.predict(X_val)
            val_acc = 100*np.sum(y_val_pred==y_val)/y_val.size

            for i in params[0]:
                self.params_[i].append(params[0][i])

            for i in params[1]:
                self.params_[i].append(params[1][i])

            self.acc_list_.append(acc)
            self.val_acc_list_.append(val_acc)
            self.t_inv_list_.append(1/(et-st))

        for i in self.params_:
            self.cv_results_[i] = self.params_[i]

        self.cv_results_["accuracy"] = self.acc_list_
        self.cv_results_["val_accuracy"] = self.val_acc_list_
        self.cv_results_["sum_accuracy"] = self.cv_results_["accuracy"] + self.cv_results_["val_accuracy"]
        self.cv_results_["t_inv"] = self.t_inv_list_
        self.cv_results_ = self.cv_results_.sort_values(by=["val_accuracy", "accuracy", "sum_accuracy", "t_inv"], ascending=False, ignore_index=True)

        self.best_params_ = self.cv_results_.iloc[0].to_dict()
        self.best_params_["early_stopping"] = bool(self.best_params_["early_stopping"])
        del self.best_params_["accuracy"]
        del self.best_params_["val_accuracy"]
        del self.best_params_["sum_accuracy"]
        del self.best_params_["t_inv"]
