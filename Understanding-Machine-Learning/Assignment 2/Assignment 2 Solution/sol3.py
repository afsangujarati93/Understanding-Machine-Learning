# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
import sklearn as skl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

traindata_array = []
traindata_array = np.loadtxt('../wine.train', delimiter=",", unpack=False)
testdata_array = []
testdata_array = np.loadtxt('../wine.test', delimiter=",", unpack=False)

traindata_features = np.delete(traindata_array, 0, axis =1)
traindata_class = traindata_array[:,0]

testdata_features = np.delete(testdata_array, 0, axis =1)
testdata_class = testdata_array[:,0]

#cheatsheet to decide which algorithm to use
#http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

SVC = svm.SVC(kernel = 'linear', C=1)
SVC.fit(traindata_features, traindata_class)

scores_wine = skl.model_selection.cross_val_score(SVC, traindata_features, traindata_class, cv=10)
print('\nSVC Wine Scores for 10 fold cross validation:\n',scores_wine)
print("SVC 10 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_wine.mean(), scores_wine.std()*2))

#Accuracy too low with cross validation
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(traindata_features, traindata_class)
scores_wine = skl.model_selection.cross_val_score(neigh, traindata_features, traindata_class, cv=10)
print('\nK NEAREST Wine Scores for 10 fold cross validation:\n',scores_wine)
print("K NEAREST 10 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_wine.mean(), scores_wine.std()*2))


RF = RandomForestClassifier(n_estimators = 50)
RF.fit(traindata_features, traindata_class)
scores_wine = skl.model_selection.cross_val_score(RF, traindata_features, traindata_class, cv=10)
print('\nRandom Forest Wine Scores for 10 fold cross validation:\n',scores_wine)
print("Random Forest 10 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_wine.mean(), scores_wine.std()*2))

#Better used for deep learning problems like image recognition
MLP = MLPClassifier(hidden_layer_sizes = (10,20,10), max_iter = 10)
MLP.fit(traindata_features, traindata_class)
scores_wine = skl.model_selection.cross_val_score(RF, traindata_features, traindata_class, cv=10)
print('\nMLP Scores for 10 fold cross validation:\n',scores_wine)
print("MLP 10 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_wine.mean(), scores_wine.std()*2))

#considering the accuracy of SVM on basis of K fold validation
predicted_class_RF = RF.predict(testdata_features) 