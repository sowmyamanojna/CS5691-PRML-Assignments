import missingno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor

###################################################################
class PreProcess():
    def __init__(self):
        pass

    def clean(self, df, verbose=False):
        if verbose==True:
            print("="*60)
        if verbose==True:
            print("Sample of Dataset")
        if verbose==True:
            print(df.head())

        if verbose==True:
            print("="*60)
        if verbose==True:
            print("Information of Dataset")
        df.info()

        if verbose==True:
            print("="*60)
        if verbose==True:
            print("'Nan'  Distribution:")
        if verbose==True:
            print(df.isnull().sum())

        if verbose==True:
            print("="*60)
        if verbose==True:
            print("Description of the dataset:")
        if verbose==True:
            print(df.describe())

        plt.figure()
        missingno.matrix(df)
        plt.title("Missing Values Visualization")
        plt.show()

        if verbose==True:
            print("Removing the Rows with NaNs")
        df_clean = df.dropna(axis=0)
        plt.figure()
        missingno.matrix(df_clean)
        plt.title("Missing Values Visualization - After NaN removal")

        df = df_clean
        desc = df.describe()
        sum(desc.loc["std"] == 0)

        # sns.pairplot(df)

        plt.figure(figsize=[15,15])
        sns.heatmap(df.corr().round(2), linewidths=.5, annot=True)
        plt.title("Heatmap of the data")
        plt.show()

        if verbose==True:
            print("="*60)
        if verbose==True:
            print("Identifying highly correlated features... ")
        # Use the upper triangle to mask the correlation matrix
        df_new = df.copy()
        df_new = df_new.drop(["Next_Tmin", "Next_Tmax"], axis=1)
        upper_triangle = np.triu(np.ones(df_new.corr().shape)).astype(bool)

        # Get the correlation pairs
        correlation_pairs = df_new.corr().mask(upper_triangle).abs().unstack().sort_values(ascending=False)
        correlation_pairs = pd.DataFrame(correlation_pairs)
        if verbose==True:
            print("Correlation between Features")
        if verbose==True:
            print(correlation_pairs.head(25))
        if verbose==True:
            print("="*50)

        # if verbose==True:
        #   print the highly correlated features
        highly_correlated = correlation_pairs[correlation_pairs[0]>0.75]
        if verbose==True:
            print("Highly Correlated Features")
        if verbose==True:
            print(highly_correlated)
        if verbose==True:
            print("="*50)

        # Remove each of the highly correlated features and check
        highly_correlated.reset_index(inplace=True)
        hc_features = highly_correlated["level_0"]

        df_new.head()
        if verbose==True:
            print("Done!")

        # Remove highly correlated features
        for feature in hc_features:
            if verbose==True:
                print("Removing:", feature, "...")
            df_new = df_new.drop([feature], axis=1)
            upper_triangle = np.triu(np.ones(df_new.corr().shape)).astype(bool)
            c = df_new.corr().mask(upper_triangle).abs().unstack().sort_values(ascending=False)
            c = pd.DataFrame(c)
            if verbose==True:
                print(c[c[0]>0.75])
            if verbose==True:
                print("="*50, "\n")
            if c[c[0]>0.75].size == 0:
                if verbose==True:
                    print("The number of highly  correlated fetaures has become zero!")
                if verbose==True:
                    print("Preventing all further removals :)")
                break

        df_new.head()

        if verbose==True:
            print("="*60)
        if verbose==True:
            print("Checking for Variance Inflation Factor... ")
        # Variance Inflation Factor - directly correlated
        X_df = df_new.copy()

        vif = pd.DataFrame()
        vif["Features"] = X_df.columns
        vif["VIF Factor"]=[variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
        vif.sort_values(by=["VIF Factor"], ascending=False, inplace=True)

        max_col = vif[vif["VIF Factor"]>1000]
        for feature in max_col["Features"]:
            max_col = vif[vif["VIF Factor"]>1000]
            if verbose==True:
                print("Features with high VIF:")
            if verbose==True:
                print(max_col)
            if verbose==True:
                print("Dropping", feature, "...")
            X_df.drop([feature], axis=1, inplace=True)

            vif = pd.DataFrame()
            vif["Features"] = X_df.columns
            vif["VIF Factor"]=[variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
            vif.sort_values(by=["VIF Factor"], ascending=False, inplace=True)
            
            if vif[vif["VIF Factor"]>1000].size==0:
                if verbose==True:
                    print("The number of fetaures that have high VIF has become zero!")
                if verbose==True:
                    print("Preventing all further removals :)")
                break
        if verbose==True:
            print("Done!")

        df_new = X_df
        df_new.head()

        df_save = df_new.copy()
        df_save["Next_Tmin"] = df["Next_Tmin"]
        df_save["Next_Tmax"] = df["Next_Tmax"]
        df_save.to_csv("../datasets/processed.csv")
        # df_save.to_csv("processed.csv")
        if verbose==True:
            print("Saved the data as processed.csv")

        return 1