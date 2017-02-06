
import numpy as np
import pickle
import os
import cPickle
import random




label = { 0: [1,0,0,0,0,0,0,0,0,0], 1 : [0,1,0,0,0,0,0,0,0,0], 2: [0,0,1,0,0,0,0,0,0,0], 3 : [0,0,0,1,0,0,0,0,0,0], 4: [0,0,0,0,1,0,0,0,0,0], 5 : [0,0,0,0,0,1,0,0,0,0],6: [0,0,0,0,0,0,1,0,0,0], 7 : [0,0,0,0,0,0,0,1,0,0],8: [0,0,0,0,0,0,0,0,1,0], 9 : [0,0,0,0,0,0,0,0,0,1]}


def unpickle(file):
 
    fo = open(file, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d
   
def tograyscale(image):

    fs =[]
    k = 0

    for i in range(0, 1024):

       fs.append((image[i] + image[i + 1024] + image[i + 2048]) / 3)
       
    return fs

def getFeatureSet(d):
    # features is a 10000x3072 vector               
    # Each row of the array stores a 32x32 colour image.
    # The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
    #Normalize values between 0 and 1

    #a = list(list)

    fs = tograyscale(d['data'])

    features = [x / 255.0 for x in fs]

    #labels is  a list of 10000 numbers in the range 0-9. 
    #The number at index i indicates the label of the ith image in the array data.
    labels = d['labels']
    
    feature_set =[]

    #for each vector of numeric features
    for feature_vect in features:
        #vector of label
        feature_label = label[labels.pop(0)]

        #Format for tensor flow input :[[[f0,f1,...],[label]],[[f0,f1,...],[label]],...]
        feature_line =[feature_vect,feature_label]
        feature_set.append(feature_line)
    

    return feature_set

def createFeatureSet():
    training_features = getFeatureSet(unpickle('cifar-10-batches-py/data_batch_1')) + getFeatureSet(unpickle('cifar-10-batches-py/data_batch_2'))+ getFeatureSet(unpickle('cifar-10-batches-py/data_batch_3'))+ getFeatureSet(unpickle('cifar-10-batches-py/data_batch_4'))+ getFeatureSet(unpickle('cifar-10-batches-py/data_batch_5'))
    test_features = getFeatureSet(unpickle('cifar-10-batches-py/test_batch'))

    #Shuffle
    random.shuffle(training_features)
    
    training_features = np.array(training_features)
    test_features = np.array(test_features)

    train_feat = list(training_features[:,0])
    train_label = list(training_features[:,1])
    test_feat = list(test_features[:,0])
    test_label = list(test_features[:,1])
    
    return train_feat,train_label,test_feat,test_label
  

if __name__ == "__main__":

    #getFeatureSet(unpickle('cifar-10-batches-py/data_batch_1'))

    createFeatureSet()
  