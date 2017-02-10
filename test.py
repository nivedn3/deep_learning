


import numpy as np 
import random

class Network(object):
	def __init__(self,size):
		self.biases=[np.random.randn(y,1) for y in size]
		self.weights=[np.random.randn(x,y) for (x,y) in zip(size[:-1],size[1:])]
		self.num_layers=len(size)

## Calculating the mini gradient descent
	def gradient_descend(self,training_data,test_data,epochs,mini_batch_length):
		if training_data:
			n_train=len(training_data)
		n=len(test_data)
		for i in range(epochs):	
			random.shuffle(training_data)
			mini_batches=[training_data[w,w+mini_batch_length] for w in xrange(0,n,mini_batch_length)]
			for i in mini_batches:
				self.update(mini_batches)
# updating the parameters
	def update(self,mini_batches):
		deriv_a=[]
		deriv_b=[]
		for i in mini_batches:
			del_b,del_w = self.backprop(i)
			l=0
			for w,b in zip(self.weights,self.biases):
				w = w - eta/len(mini_batches)*del_w[l]
				b = b - eta/len(mini_batches)*del_b[l]
				l=l+1
			

	def feedforward(self,a):	
		for b,w in zip(self.biases,self.weights):
			a=np.dot(w,a)+b
		return a 
	def backprop(self,test_data):
		delta=[]
		z=[]
		del_a=[]
		del_b=[]
		a=training_data
		for w,b in zip(self.weights,self.biases):
			z.append(np.dot(a.w)+b)
			a=sig(z)
		deltaL=np.dot((sig(feedfor(training_data))-training_data),sig_prime(z[:-1]))
		delta.append(deltaL)
		del_a.append(np.dot(deltaL,sig(z[:-2])))

		for l in range(len(size)-1):
			delta.append(np.transpose(np.dot(self.weights[-l-1])*delta[-l],sig_prime(z[-l-2])))
			del_a.append(np.dot(delta[:-l-1],sig(z[:-l-2])))
		return del_b=delta[::-1]
		return del_a=del_a[::-1]
		
	def sig(self,n):
		return 1/(1+np.exp(-n))

	def sig_prime(self,n):
		return sig(n)*(1-sig(n))
