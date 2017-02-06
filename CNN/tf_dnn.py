from tf_input import createFeatureSet 
import tensorflow as tf

import pickle
import numpy as np

train_feat,train_label,test_feat,test_label = createFeatureSet() 


n_labels = 10

#Batch size must be bigger for a bigger dataset
batch_size = 100

#Training iterations
iterations = 100

x = tf.placeholder('float')
y = tf.placeholder('float')


keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
def conv2d(x, W):
	return tf.nn.conv2d(x, W,strides =[1,1,1,1], padding = 'SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x,ksize= [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


#Define model
def convolutional_network(x):
	#Dictionaries
	weights = {	'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])), #5x5 convolution 1 input 32 features(outputs)
				'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])), #5x5 convolution 32 input 64 features(outputs)
				'W_fc': tf.Variable(tf.random_normal([8*8*64*3,1024])),#fully connected
				'W_out': tf.Variable(tf.random_normal([1024,n_labels]))} 
	biases = {	'b_conv1': tf.Variable(tf.random_normal([32])), 
				'b_conv2': tf.Variable(tf.random_normal([64])), 
				'b_fc': tf.Variable(tf.random_normal([1024])),
				'b_out': tf.Variable(tf.random_normal([n_labels]))} 
	

	x =	 tf.reshape(x,shape = [-1,32,32,1]) #reshape images
	print(x.get_shape())
	conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)
	print(conv1.get_shape())
	conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2'])+ biases['b_conv2'])
	conv2 = maxpool2d(conv2)
	print(conv2.get_shape())
	fc = tf.reshape(conv2,[-1,8*8*64*3])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

	fc = tf.nn.dropout(fc,keep_rate)

	print(fc.get_shape())
	output = tf.matmul(fc, weights['W_out'])+biases['b_out']
	print(output.get_shape())

	return output

#Train neural network

def train_neural_network(x):
	
	#Take input data and pass through NN model.
	prediction = convolutional_network(x)
	#Calculate the difference of the prediction and label
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	#Back propagation
	#Learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost) 

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
	    
		for iteration in range(iterations):
			loss = 0
			i=0
			while i < len(train_feat):
				start = i
				end = i+batch_size
				batch_feat = np.array(train_feat[start:end])
				batch_label = np.array(train_label[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_feat,
				                                              y: batch_label})
				loss += c
				i+=batch_size
				
			print('Iteration', iteration+1,'loss:',loss)
		
		#Compare the prediction to the label
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		#Evaluate all the accuracy
		print('Accuracy:',accuracy.eval({x:test_feat, y:test_label}))



train_neural_network(x)