#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
# features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import tree
from sklearn.metrics import accuracy_score
from time import time

for percentile in [1,10]:
    print("PERCENTILE = %d"%percentile)
    features_train, features_test, labels_train, labels_test = preprocess(percentile=percentile)

    print("n. of features: %d"%len(features_test[0]))
    t = tree.DecisionTreeClassifier(min_samples_split=40)
    t0 = time()
    clf = t.fit(features_train,labels_train)
    print ("trained, training time: %fs"%(round(time()-t0, 3)))
    pred = clf.predict(features_test)
    print ("accuracy_score: %f\n"%accuracy_score(pred,labels_test))

#########################################################


