


import numpy as np 
import random

class Network(object):
	def __init__(self,size):
		self.biases=[np.random.randn(y,1) for y in size[1:]]
		self.weights=[np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
		self.num_layers=len(size)

## Calculating the mini gradient descent
	def gradient_descend(self,training_data,test_data,epochs,mini_batch_length,eta):
		if test_data:
			n_test=len(test_data)
		n=len(training_data)
		print n
		for j in xrange(epochs):	
			random.shuffle(training_data)
			mini_batches=[training_data[w:w+mini_batch_length] for w in xrange(0,n,mini_batch_length)]

			for i in mini_batches:
				self.update(i,eta)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)
# updating the parameters
	def update(self,mini_batches,eta):
		derb=[np.zeros(b.shape) for b in self.biases]
		derw=[np.zeros(w.shape) for w in self.weights]

		for i,j in mini_batches:
			del_w,	del_b = self.backprop(i,j)
			derw=[w+sw for w,sw in zip(derw,del_w)]
			derb=[b+sb for b,sb in zip(derb,del_b)]
			#print j
			#print del_w[0].shape
			

			
		self.weights = [w - (eta/len(mini_batches))*p for w,p in zip(self.weights,derw)]
		self.biases = [b - (eta/len(mini_batches))*c for b,c in zip(self.biases,derb)]
			


	def evaluate(self,test_data):
		test_r=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
		return sum(int(x == y) for (x, y) in test_r)

	def feedforward(self,a):	
		for b,w in zip(self.biases,self.weights):
			a=sig(np.dot(w,a)+b)
		return a 
	def backprop(self,i,j):	
		delta=[]
		z=[]
		del_a=[]
		del_b=[]
		a=i
		count=0

		for w,b in zip(self.weights,self.biases):	
			#print w.shape
			#print "b", b.shape
			z.append(np.dot(w,a)+b)
			a=sig(z[count])
			count=count+1
		

		deltaL=((sig(z[1])-j)*sig_prime(z[1]))
		delta.append(deltaL)
		#print deltaL.shape
		#print z[-2].shape
		del_a.append(np.dot(deltaL,np.transpose(sig(z[0]))))

		for l in range(self.num_layers-2):
		
			zeta=(sig_prime(z[-l-2])*np.dot(np.transpose(self.weights[-l-1]),delta[-1]))
			delta.append(zeta)
			del_a.append(np.dot(delta[-1],np.transpose(i)))
		del_b=delta[::-1]
		del_a=del_a[::-1]
		#print del_a
		return del_a,del_b
		

def sig(n):
	return 1.00000/(1.00000+np.exp([-x for x in n]))

def sig_prime(n):
	return sig(n)*(1-sig(n))



if __name__=='__main__':
	import sys
	sys.path.insert(0,'/home/nived/mnist/nn/src')
	import mnist_loader
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	net=Network([784,30,10])
	net.gradient_descend(training_data,test_data,30,25,3.00000000000)
