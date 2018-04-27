#!/usr/bin/python2


####################################
####################################
#           USE PYTHON 2           #
####################################
####################################
"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
clf = dtc.fit(features,labels)
pred = clf.predict(features)

from sklearn.metrics import accuracy_score

print("accuracy using the entire dataset for both training and testing: %.4f (very overfitted)"%accuracy_score(labels,pred))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.3, random_state=42)

clf = dtc.fit(X_train,y_train)
pred = clf.predict(X_test)
print ("accuracy using cross validation with 30/100 for testing: %.4f "%accuracy_score(pred,y_test))

from collections import Counter
print ("Test data labels: " + str(Counter(y_test)))

from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
print ("Confusion matrix using cross validation with 30/100 for testing:")
pt = PrettyTable()
tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
pt.add_row(["True -> \n Predicted \\|/","Poi","!Poi"])
pt.add_row(["Poi",tp,fp])
pt.add_row(["!Poi",fn,tn])
print (pt)

print ("POI precision: %.4f"%(tp/(tp+fp)))
print ("POI recall: %.4f"%(tp/(tp+fn)))
print("Predictions vs True:")
pt = PrettyTable()
pt.add_column("Predictions", pred)
pt.add_column("True",y_test)
#print (pt)