import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
import numpy as np

x = tf.placeholder(tf.float32,shape = [None,784])
#y = tf.placeholder(tf.float32,shape = [None,10])
g_in = tf.placeholder(tf.float32,shape = [None,128])

def weight(shape):

	values = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(values)

dw1 = weight([784,128])
db1 = weight([128])

dw2 = weight([128,1])
db2 = weight([1])

vard = [dw1,dw2,db1,db2]

def prob(k):
	d_fc1 = tf.nn.relu(tf.matmul(k,dw1) + db1)
	d_fc2 = tf.nn.sigmoid(tf.matmul(d_fc1,dw2) + db2)
	return d_fc2

gw1 = weight([128,784])
gb1 = weight([784])

g_fc = tf.nn.sigmoid(tf.matmul(g_in,gw1) + gb1)
g_out = tf.reshape(g_fc,[50,28,28,1])
tf.summary.image('input',g_out,3)

varg = [gb1,gw1]

d = prob(x) 
g = prob(g_fc)

with tf.name_scope("dloss"):
	dloss = -tf.reduce_mean(tf.log(d) + tf.log(1-g))
	tf.summary.scalar('dloss',dloss)

with tf.name_scope("gloss"):
	gloss = -tf.reduce_mean(tf.log(g))
	tf.summary.scalar('gloss',gloss)

dtrain = tf.train.AdamOptimizer().minimize(dloss, var_list=vard)
gtrain = tf.train.AdamOptimizer().minimize(gloss, var_list=varg)
sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/psycholearner/projects/deeplearning/deep_learning/5")
writer.add_graph(sess.graph)

for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%3 == 0:
		s = sess.run(merged_summary,feed_dict = {x: batch[0],g_in: np.random.uniform(-1., 1., size=[50,128])})
		writer.add_summary(s,i)

	sess.run([dtrain, dloss], feed_dict={x: batch[0], g_in: np.random.uniform(-1., 1., size=[50,128])})
	sess.run([gtrain, gloss], feed_dict={g_in: np.random.uniform(-1., 1., size=[50,128])})