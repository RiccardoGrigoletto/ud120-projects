#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# reducing the training set of 99%
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 


from sklearn import svm
from sklearn.metrics import accuracy_score
C = [10.,100.,1000.,10000.]
# C = [10000.]
pred = []
for c in C:
    clf = svm.SVC(C=c, kernel='rbf')
    clf.fit(features_train,labels_train)
    predTmp = clf.predict(features_test)
    pred.append(predTmp)

    acc = accuracy_score(predTmp, labels_test)
    print("C: %d , accuracy: %f"%(c,acc))

email_results = [10,26,50]
for i in email_results:
    for index,p in enumerate(pred, start=0):
        print("prediction: %d,\tC: %d ,\tthe %dth email (0 based index) come from: %s"%(index,C[index],i,("chris" if p[i]==1 else "sara")))
sara_emails = 0
chris_emails = 0
for pIndex,p in enumerate(pred, start=0):
    for index,i in enumerate(p, start=0):
        if (i==0): sara_emails+=1
        else: chris_emails+=1
    print("prediction: %d, C: %d"%(pIndex,C[pIndex]))
    print("\tchris emails: %d"%chris_emails)
    print("\tsara emails: %d"%sara_emails)
    sara_emails = 0
    chris_emails = 0




#########################################################

