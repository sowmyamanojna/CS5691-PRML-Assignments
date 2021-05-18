import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression import PolynomialRegression

class GridSearch():
    def __init__(self):
        pass

    def get_result(self, df, sample_size, degrees_allowed, lmbda_list):
        df_sample = df.sample(n=sample_size, random_state=42)

        df_train = df_sample.sample(frac=0.9, random_state=42)
        df_val = df_sample[~df_sample.index.isin(df_train.index)]

        X_train = df_train["x"].to_numpy().reshape(-1, 1)
        X_val = df_val["x"].to_numpy().reshape(-1, 1)

        y_train = df_train["y"].to_numpy().reshape(-1, 1)
        y_val = df_val["y"].to_numpy().reshape(-1, 1)
        
        self.result = []
        self.correspondance = {}
        
        for degree in degrees_allowed:
            for lmbda in lmbda_list:
                regressor = PolynomialRegression()
                regressor.fit(X_train, y_train, degree=degree, lmbda=lmbda)
                y_train_pred = regressor.transform(X_train)
                y_val_pred = regressor.transform(X_val)

                train_error = regressor.error(y_train, y_train_pred)
                val_error = regressor.error(y_val, y_val_pred)

                self.result.append([degree, lmbda, train_error, val_error])
                self.correspondance[(degree, lmbda)] = {"df_sample":df_sample, "regressor":regressor}

        df_results = pd.DataFrame(self.result, columns=["degree", "lambda", "Train error", "Validation error"])
        df_results["Sum Error"] = df_results["Train error"] + df_results["Validation error"]
        df_results.sort_values(by="Sum Error", inplace=True)
        return df_results, self.correspondance

    def get_plots(self, X, y, correspondance, sample_size, show):
        for key in correspondance:
            df_sample = correspondance[key]["df_sample"]
            df_sample.sort_values(by=["x"], inplace=True)
            X_sample = df_sample["x"].to_numpy().reshape(-1,1)
            y_sample = df_sample["y"].to_numpy().reshape(-1,1)

            regressor = correspondance[key]["regressor"]
            y_pred_sample = regressor.transform(X_sample)

            title = "Curve Fitting - Degree: "+str(regressor.degree)\
            +"; Sample Size: "+str(sample_size)+"; $\lambda$: "\
            +str(regressor.lmbda)
            fname = "d_"+str(regressor.degree)+"_size_"+str(sample_size)+"_l_"+str(regressor.lmbda)+".png"
            
            plt.figure()
            plt.plot(X, y, label="True Value")
            if y_sample.size >= 100:
                plt.plot(X_sample, y_sample, 'r.', alpha=0.5, label="Sampled points")
            else:
                plt.plot(X_sample, y_sample, 'ro', alpha=0.75, label="Sampled points")
            plt.plot(X_sample, y_pred_sample, label="Predicted Value")
            if title:
                plt.title(title)
            plt.xlabel("X-values")
            plt.ylabel("Y-values")
            plt.legend()
            plt.savefig("images/"+fname)
            if show:
                plt.show()
