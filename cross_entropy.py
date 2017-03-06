import numpy as np 
import random
import json


class crossentropy(object):
	@staticmethod
	def cross_cost(a,y):
		return np.sum(-(y*np.log(a)+(1-y)*np.log(1-a)))

	@staticmethod
	def derivative:
		return a-y

class Network(object):
	