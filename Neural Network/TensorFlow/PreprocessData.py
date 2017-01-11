import numpy as np
import random
import pickle

#Labels are represented as vector
label = {'unacc': [1,0,0,0], 'acc': [0,1,0,0], 'good': [0,0,1,0], 'vgood':[0,0,0,1]}

#Dictionary to convert features into numeric representation 
buying = {'low': 0, 'med': 1, 'high': 2, 'vhigh':3}
maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh':3}
doors = {'2': 2, '3': 3, '4': 4, '5more':5}
person = {'2': 2, '4': 4, 'more': 6}
lugboot = {'small': 0, 'med': 1, 'big': 2}
safety = {'low': 0, 'med': 1, 'high': 2}


def read_file(doc):
	feature_set = []
	fs = []
	#read line by line and store it in a vector
	with open(doc,'r') as f:
		fs+=[line.split() for line in f] 
	for line in fs:
		#vector of numeric features
		feature_vect = [buying[line[0]],maint[line[1]],doors[line[2]],person[line[3]],lugboot[line[4]],safety[line[5]]]
		#vector of label representation
		feature_label = label[line[6]]
		#Format for tensor flow input :[[[f0,f1,...],[label]],[[f0,f1,...],[label]],...]
		feature_line =[feature_vect,feature_label]
		feature_set.append(feature_line)
	
	return feature_set



def create_feature_set(dataset,test_size = 0.1):
	features = read_file('car.data')
	random.shuffle(features)
	features = np.array(features)
	
	testing_size = int(test_size*len(features))

	#train feat has 90% of features vectors
	train_feat = list(features[:,0][:-testing_size])
	#train label has 90% of label vectors
	train_label = list(features[:,1][:-testing_size])
	#test feat has 10% of features vectors
	test_feat = list(features[:,0][-testing_size:])
	#test label has 10% of label vectors
	test_label = list(features[:,1][-testing_size:])


	return train_feat,train_label,test_feat,test_label



if __name__ == '__main__':
	train_feat,train_label,test_feat,test_label= create_feature_set('car.data')
	# if you want to pickle this data:
	with open('car_set.pickle','wb') as f:
			pickle.dump([train_feat,train_label,test_feat,test_label],f)

			