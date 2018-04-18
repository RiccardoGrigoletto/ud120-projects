#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.savefig("data.png")
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

for algo in ["adaboost","random_forest","KNN"]:
    clf = 0
    if algo == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier().fit(features_train, labels_train)
    if algo == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier().fit(features_train, labels_train)
    if algo == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=8).fit(features_train, labels_train)
    try:
        prettyPicture(clf, features_test, labels_test,name=algo)
    except NameError:
        pass
    from sklearn.metrics import accuracy_score
    print("%s accuracy: %f"%(algo,accuracy_score(labels_test,clf.predict(features_test))))

# for n_estimators in [50,100,150,200,250,300,350,400,460,500]:
#     clf = 0
#     from sklearn.ensemble import AdaBoostClassifier
#     clf = AdaBoostClassifier(n_estimators=n_estimators).fit(features_train, labels_train)
#     print("%d adaBoost accuracy: %f"%(n_estimators,accuracy_score(labels_test,clf.predict(features_test))))

maxAccuracy = 0
for k in range(2,200):
    clf = 0
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k).fit(features_train, labels_train)
    accScore = accuracy_score(labels_test,clf.predict(features_test))
    print("%d nearest neighbors accuracy: %f"%(k,accScore))
    if (maxAccuracy < accScore): maxAccuracy = accScore

print(maxAccuracy)