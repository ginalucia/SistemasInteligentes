from PreprocessData import create_feature_set 
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

import pickle
import numpy as np

#Obtain numeric representation fron the dataset car.data
#train = 90% test = 10%
train_feat,train_label,test_feat,test_label = create_feature_set('car.data') 

#Model is designed with 3 hidden layers 
#Number of nodes per layer 
n_nodes_hidden_layer_1 = 50
n_nodes_hidden_layer_2 = 50
n_nodes_hidden_layer_3 = 50

#Number of output labels :{'unacc': [1,0,0,0], 'acc': [0,1,0,0], 'good': [0,0,1,0], 'vgood':[0,0,0,1]
n_labels = 4

#Batch size must be bigger for a bigger dataset
batch_size = 1

#Training iterations
iterations = 50

x = tf.placeholder('float')
y = tf.placeholder('float')

#Initialize layers with random values from a normal distribution of shape [a,b] -> a : size of input,  b : number of nodes in layer 

hidden_1_layer = {	'weights': tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hidden_layer_1])),
					'bias':tf.Variable(tf.random_normal([n_nodes_hidden_layer_1]))}

hidden_2_layer = {	'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_l,n_nodes_hidden_layer_2])),
					'bias':tf.Variable(tf.random_normal([n_nodes_hidden_layer_2]))}
	
hidden_3_layer = {	'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_2,n_nodes_hidden_layer_3])),
					'bias':tf.Variable(tf.random_normal([n_nodes_hidden_layer_3]))}

output_layer = {	'weights': tf.Variable(tf.random_normal([n_nodes_hidden_layer_3,n_labels])),
					'bias':tf.Variable(tf.random_normal([n_labels]))}

#Define model
def neural_network_model(data):
	
	#For each layer, calculate (input*weights)+bias
	
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1) #activation function
	
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2) #activation function
	
	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['bias'])
	l3 = tf.nn.relu(l3) #activation function

	output = tf.matmul(l3,output_layer['weights'])+output_layer['bias']
	
	return output

#Train neural network

def train_neural_network(x):
	
	#Take input data and pass through NN model.
	prediction = neural_network_model(x)
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



