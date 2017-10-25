# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:23:16 2017

@author: afsan
"""

from sklearn.datasets import load_iris
from sklearn import svm
import sklearn.metrics as sklm
import sklearn as skl
import numpy as np
#
iris  = load_iris()

x = iris.data
y = iris.target

svc_iris = svm.SVC(kernel = 'linear', C=1)
svc_iris.fit(x,y)

x_sepal = iris.data[:,:2]
svc_sepal = svm.SVC(kernel = 'linear', C=1)
svc_sepal.fit(x_sepal,y)

x_petal = iris.data[:,2:]
svc_petal = svm.SVC(kernel = 'linear', C=1)
svc_petal.fit(x_petal,y)

predicted_iris = svc_iris.predict(x)
predicted_sepal = svc_sepal.predict(x_sepal)
predicted_petal = svc_petal.predict(x_petal)

#plt.plot(x_sepal[:,0],x_sepal[:,1],'xr')

#what's fit here?
#confirm how does predicts, predicts the data. 
#Is it completely independent of the target data i.e. the Y data?

#Part 1.2
#>>Iris Data Metrics 
iris_accuracy = sklm.accuracy_score(y, predicted_iris)

#**what is micro and stuff what are these precision and stuff
iris_precision = sklm.precision_score(y, predicted_iris,average='micro')
iris_recall = sklm.recall_score(y, predicted_iris,average='micro')
iris_f1_score = sklm.f1_score(y, predicted_iris,average='micro')
iris_confusion_matrix = sklm.confusion_matrix(y,predicted_iris)
print('\nIris Accuracy:',iris_accuracy,'\nIris Precision:',iris_precision,'\nIris Recall:',iris_recall,'\nIris f1 score:',iris_f1_score)
print('\nIris confusion matrix:\n',iris_confusion_matrix)

#
iris_sepal_accuracy = sklm.accuracy_score(y, predicted_sepal)
iris_sepal_precision = sklm.precision_score(y, predicted_sepal,average='micro')
iris_sepal_recall = sklm.recall_score(y, predicted_sepal,average='micro')
iris_sepal_f1_score = sklm.f1_score(y, predicted_sepal,average='micro')
iris_sepal_confusion_matrix = sklm.confusion_matrix(y,predicted_sepal)
print('\nIris Sepal Accuracy:',iris_sepal_accuracy,'\nIris Sepal Precision:',iris_sepal_precision,'\nIris Sepal Recall:',iris_sepal_recall,'\nIris Sepal f1 score:',iris_sepal_f1_score)
print('\nIris Sepal confusion matrix:\n',iris_sepal_confusion_matrix)


iris_petal_accuracy = sklm.accuracy_score(y, predicted_petal)
iris_petal_precision = sklm.precision_score(y, predicted_petal,average='micro')
iris_petal_recall = sklm.recall_score(y, predicted_petal,average='micro')
iris_petal_f1_score = sklm.f1_score(y, predicted_petal,average='micro')
iris_petal_confusion_matrix = sklm.confusion_matrix(y,predicted_sepal)
print('\nIris Petal Accuracy:',iris_petal_accuracy,'\nIris Petal Precision:',iris_petal_precision,'\nIris Petal Recall:',iris_petal_recall,'\nIris Petal f1 score:',iris_petal_f1_score)
print('\nIris Petal confusion matrix:\n',iris_petal_confusion_matrix)


#Part 1.3
#>>10 fold cross validation
scores_iris = skl.model_selection.cross_val_score(svc_iris, iris.data, iris.target, cv=10)
print('\nIris Scores for 10 fold cross validation:\n',scores_iris)
print("10 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_iris.mean(), scores_iris.std()*2))

scores_iris = skl.model_selection.cross_val_score(svc_iris, iris.data, iris.target, cv=5)
print('\nIris Scores for 5 fold  cross validation:\n',scores_iris)
print("5 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_iris.mean(), scores_iris.std()*2))

scores_iris_sepal = skl.model_selection.cross_val_score(svc_sepal, x_sepal, iris.target, cv=10)
print('\nIris Sepal Scores for 10 fold  cross validation:\n',scores_iris_sepal)
print("10 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_iris_sepal.mean(), scores_iris_sepal.std()*2))

scores_iris_sepal = skl.model_selection.cross_val_score(svc_sepal, x_sepal, iris.target, cv=5)
print('\nIris Sepal Scores for 5 fold cross validation:\n',scores_iris_sepal)
print("5 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_iris_sepal.mean(), scores_iris_sepal.std()*2))


scores_iris_petal = skl.model_selection.cross_val_score(svc_sepal, x_petal, iris.target, cv=10)
print('\nIris Petal Scores for 10 fold  cross validation:\n',scores_iris_petal)
print("10 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_iris_petal.mean(), scores_iris_petal.std()*2))

scores_iris_petal = skl.model_selection.cross_val_score(svc_sepal, x_petal, iris.target, cv=5)
print('\nIris Petal Scores for 5 fold  cross validation:\n',scores_iris_petal)
print("5 fold Accuracy: %0.2f (+/- %0.2f)" % (scores_iris_petal.mean(), scores_iris_petal.std()*2))


list_kparts = []
list_classes_kparts = []
Num_parts = 10
temp_start = 0
iris_length = len(iris.data)
perset_length = iris_length/Num_parts
for i in range(Num_parts):
    temp_end =  int(temp_start+perset_length)
    temp_partarray = iris.data[temp_start:temp_end,:]
    temp_class_partarray = iris.target[temp_start:temp_end]
    temp_start = temp_end
    list_kparts.append(temp_partarray)
    list_classes_kparts.append(temp_class_partarray)

list_startKparts = []
list_uptoKparts = []
accuracy_mean_manual = 0
for i in range(Num_parts):
    
    validation_list = list_kparts[i]
    if(i==0):
        training_list = np.concatenate((list_kparts[i+1:Num_parts] ),axis = 0)
        training_class_list = np.concatenate((list_classes_kparts[i+1:Num_parts]),axis = 0)
    elif (i == (Num_parts-1)):
        training_list = np.concatenate((list_kparts[0:i]),axis = 0)
        training_class_list = np.concatenate((list_classes_kparts[0:i]),axis = 0)
    else:
        list_kpart_start = list_kparts[0:i]
        list_kparts_upto = list_kparts[i+1:Num_parts]
        
        list_class_start = list_classes_kparts[0:i]
        list_class_upto = list_classes_kparts[i+1:Num_parts]
        training_list_beforevstack = np.concatenate((list_kpart_start,list_kparts_upto ),axis = 0)
        training_class_list_beforehstack = np.concatenate((list_class_start,list_class_upto),axis = 0)
        training_list = np.vstack((training_list_beforevstack))
        training_class_list = np.hstack((training_class_list_beforehstack))
        
    svc_iris = svm.SVC(kernel = 'linear', C=1)
    svc_iris.fit(training_list,training_class_list)
    predicted_iris = svc_iris.predict(validation_list)
    iris_accuracy = sklm.accuracy_score(list_classes_kparts[i], predicted_iris)
    print('Accuracy for ', i, ' :', iris_accuracy)
    accuracy_mean_manual += iris_accuracy
    
overall_mean = accuracy_mean_manual/Num_parts
print('Mean of accuracy for manual calc: ',overall_mean )