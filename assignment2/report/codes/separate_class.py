#import numpy as np
import pandas as pd

class Separate():
	def __init__(self, data):
		self.full_data = data
		self.feat_length = data.shape[1] - 1
		self.classes = data.iloc[:,-1].unique()
		self.groups = data.groupby(self.feat_length)
		
		
	def get_y(self):
		Y = []
		for i in self.classes:
			#print(self.groups.get_group(i))
			grp = self.groups.get_group(i)
			Y.append(grp.iloc[:,self.feat_length])
		return(Y)
	def get_x(self):
		X = []
		for i in self.classes:
			grp = self.groups.get_group(i)
			X.append(grp.iloc[:,:self.feat_length])
		return(X)

	def get_separated_data(self):
		C = []
		for i in self.classes:
			#print(self.groups.get_group(i))
			grp = self.groups.get_group(i)
			C.append(grp)
		return(C)
