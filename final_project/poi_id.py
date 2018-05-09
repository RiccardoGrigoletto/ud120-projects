#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','email_address','from_poi_to_this_person']
features_list = ['poi','salary','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person'] # You will need to use more features
financial_features_list = ['poi','salary','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','restricted_stock_deferred','total_stock_value','expenses','loan_advances','other','director_fees','deferred_income','long_term_incentive','shared_receipt_with_poi']
email_features_list = ['poi','to_messages','shared_receipt_with_poi','from_messages','from_this_person_to_poi','from_poi_to_this_person'] #'email_address'

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
def deleteOutliers(dictionary,financial_data,contamination=0.02):
    from feature_format import targetFeatureSplit
    financial_labels, financlial_features = targetFeatureSplit(financial_data)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(contamination=contamination)
    print("IsolationForest")
    for key,out in zip(dictionary.keys(),clf.fit_predict(financlial_features,financial_labels)):
        if (out==-1):
            print("REMOVING: %s: "%key)
            del dictionary[key]
    return dictionary

data_dict.pop("TOTAL") # TOTAL is not a person, it's the sum of the features
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict = deleteOutliers(data_dict,featureFormat(data_dict, features_list, sort_keys = True))


    
financial_data = featureFormat(data_dict, financial_features_list, sort_keys = True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
financial_data = scaler.fit_transform(financial_data)

# Using isolation forest to remove outliers, 
# split data
financial_labels, financial_features = targetFeatureSplit(financial_data)

### Task 3: Create new feature(s)
new_features = []
for index, (key, value) in enumerate(data_dict.iteritems()):
    tmp1 = -1
    tmp2 = -1
    tmp3 = -1
    tmp4 = -1
    for k1,v1 in value.iteritems():
        if k1 == 'from_messages':
            if str(v1) != v1:
                tmp1=v1
        if k1 == 'from_poi_to_this_person':
            if str(v1) != v1:
                tmp2=v1
        if k1 == 'to_messages':
            if str(v1) != v1:
                tmp3=v1  
        if k1 == 'from_this_person_to_poi':
            if str(v1) != v1:
                tmp4=v1           
    if tmp1 != -1 and tmp2 != -1:
        data_dict[key]['from_poi_ratio'] = float(tmp2)/float(tmp1)
    else: data_dict[key]['from_poi_ratio'] = 'NaN'    
    if tmp3 != -1 and tmp4 != -1:
        data_dict[key]['to_poi_ratio'] = float(tmp4)/float(tmp3)
    else: data_dict[key]['to_poi_ratio'] = 'NaN'

    t0 = data_dict[key]['poi']
    t1 = 0
    t2 = 0
    for k1,v1 in value.iteritems():
        if k1 == 'from_poi_ratio':
            if str(v1) != v1:t1 = v1 
        if k1 == 'to_poi_ratio':
            if str(v1) != v1:t2 = v1 
    new_features.append((t0,t1,t2))
financial_features_list.append('from_poi_ratio')
financial_features_list.append('to_poi_ratio')

from sklearn.model_selection import train_test_split
selected_data = featureFormat(data_dict, financial_features_list, sort_keys = True)

# Using isolation forest to remove outliers, 
# split data
selected_labels, selected_features = targetFeatureSplit(selected_data)
X_selected_train, X_selected_test, y_selected_train, y_selected_test = train_test_split(selected_features,selected_labels, test_size=0.3, random_state=42)

from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized',n_components=2)

features_to_save = 3
if (features_to_save > 0):
    pca.fit([x[:-features_to_save] for x in X_selected_train],y_selected_train)
    X_pca_train = pca.transform([x[:-features_to_save] for x in X_selected_train])
    X_pca_test = pca.transform([x[:-features_to_save] for x in X_selected_test])
else:
    pca.fit(X_selected_train,y_selected_train)
    X_pca_train = pca.transform(X_selected_train)
    X_pca_test = pca.transform(X_selected_test)
pca_components = sorted(enumerate(pca.components_[1]),key=lambda x : abs(x[1]),reverse=True)

for c in range(features_to_save):
    if c != 0:
        X_pca_train = np.c_[X_pca_train,np.zeros(len(X_pca_train))]
for i,t in enumerate(X_selected_train):
    for c in range(features_to_save):
        if c != 0:
         X_pca_train[i][-c] = t[-c]
for c in range(features_to_save):
    if c != 0:
        X_pca_test = np.c_[X_pca_test,np.zeros(len(X_pca_test))]
for i,t in enumerate(X_selected_test):
    for c in range(features_to_save):
        if c != 0:
            X_pca_test[i][-c] = t[-c]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
X_train = X_pca_train
y_train = y_selected_train
X_test = X_pca_test
y_test = y_selected_test

features_list = financial_features_list
selected_features_list = ['poi',financial_features_list[pca_components[0][0]],financial_features_list[pca_components[1][0]],'shared_receipt_with_poi','from_poi_ratio','to_poi_ratio']

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
test_classifier(clf,data_dict,selected_features_list)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(weights='distance',n_neighbors=2)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
test_classifier(clf,data_dict,selected_features_list)

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
test_classifier(clf,data_dict,selected_features_list)

from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
test_classifier(clf,data_dict,selected_features_list)

from sklearn.grid_search import GridSearchCV
param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
test_classifier(clf,data_dict,selected_features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, data_dict, selected_features_list)