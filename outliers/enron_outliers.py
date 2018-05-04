#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from outlier_removal_regression

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
target, features = targetFeatureSplit( data )
### your code below
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.LinearRegression()
reg.fit(feature_train,target_train)
pred = reg.predict(feature_test)
for point in data:
    print(point)
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
#matplotlib.pyplot.plot(feature_train, reg.predict(feature_train), color="r") 
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

