import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
class Confusion_matrix():
	def __init__(self,y_pred, y_orig):
		self.pred = y_pred
		self.original = y_orig
		self.length = len(y_pred)
		self.compare = y_pred == y_orig
		self.accuracy = np.sum(self.compare)/self.length
		self.classes = pd.Series(y_orig).unique()[0]
		
	def get_matrix(self):
		#mat = np.zeros((l,l))
		#conf_matrix = pd.crosstab(self.original,self.pred,rownames=["actual"],colnames = ["predicted"])
		mat = confusion_matrix(self.original,self.pred)
		return(mat)

		
			
		
		
		

