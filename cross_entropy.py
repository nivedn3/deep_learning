# Using the cross entropy transfer function instead of the sigmoid function used in the last case.


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
	def __init__(self,sizes,):
		self.initialise()
		self.layers=len(sizes)
		self.size=sizes

	def initialise(self):
		self.weights=  [np.random.randn(y,x)/sqrt(y)  for x,y in zip(self.size[1:],self.size[:-1])]
		self.biases= [np.random.randn(x,1) for x in self.size[1:]]

	def feedfoward(self,a):
		for w,b in zip(self.weights,self.biases):
			z=np.dot(w,a)+b
			a=sigmoid(z)
		return a 

	def sgd(self,mbsize,training_data):
		mini_batch=[training_data[k:k+mbsize]  for k in range(0,len(training_data),mbsize)]
		self.update_minibatch(mini_batch)

	def update_minibatch(self):


	def backprop(self):
		



def sigmoid(a)
	return 1/(1+np.exp(-a))

def sigmoid_prime(a):
	return  sigmoid(a)*(1-sigmoid(a))



