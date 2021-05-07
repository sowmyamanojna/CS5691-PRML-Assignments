import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.neural_network import MLPClassifier

class GridSearch():
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
        self.cv_results_ = self.cv_results_.sort_values(by=["accuracy", "sum_accuracy", "t_inv"], ascending=False, ignore_index=True)

        self.best_params_ = self.cv_results_.iloc[0].to_dict()
        del self.best_params_["accuracy"]
        del self.best_params_["val_accuracy"]
        del self.best_params_["sum_accuracy"]
        del self.best_params_["t_inv"]
        