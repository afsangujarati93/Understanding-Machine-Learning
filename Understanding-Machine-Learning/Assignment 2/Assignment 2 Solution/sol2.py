# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sklm
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

train_20news = fetch_20newsgroups(subset='train')
test_20news = fetch_20newsgroups(subset='test')
shuttled_20news = fetch_20newsgroups(subset='all', shuffle = True)
y_test = test_20news.target
y_train = train_20news.target
#TfidfVectorizer()
#Convert a collection of raw documents to a matrix of TF-IDF features.
#Equivalent to CountVectorizer followed by TfidfTransformer.

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_20news.data)
test_vectors = vectorizer.transform(test_20news.data)

#counterizer = CountVectorizer()
#test_vectors = counterizer.fit_transform(test_20news.data)
#
#transforizer = TfidfTransformer()
#test_vectors = transforizer.fit_transform(test_vectors)

RF = RandomForestClassifier(n_estimators = 50)
RF.fit(train_vectors, y_train)
predicted_vectors = RF.predict(test_vectors)

vectors_accuracy = sklm.accuracy_score(y_test, predicted_vectors)

#**what is micro and stuff what are these precision and stuff
vectors_precision = sklm.precision_score(y_test, predicted_vectors,average='micro')
vectors_recall = sklm.recall_score(y_test, predicted_vectors,average='micro')
vectors_f1_score = sklm.f1_score(y_test, predicted_vectors,average='micro')
vectors_confusion_matrix = sklm.confusion_matrix(y_test,predicted_vectors)

pipeline = Pipeline([
    ('clf', RandomForestClassifier(n_estimators = 50))
])

pipeline.fit(train_vectors, y_train)
predicted_vectors = pipeline.predict(test_vectors)

vectors_accuracy = sklm.accuracy_score(y_test, predicted_vectors)

#**what is micro and stuff what are these precision and stuff
vectors_precision = sklm.precision_score(y_test, predicted_vectors,average='macro')
vectors_recall = sklm.recall_score(y_test, predicted_vectors,average='macro')
vectors_f1_score = sklm.f1_score(y_test, predicted_vectors,average='macro')
vectors_confusion_matrix = sklm.confusion_matrix(y_test,predicted_vectors)

pipeline_mlp = Pipeline([
    ('clf', MLPClassifier(hidden_layer_sizes = (10,20,10), max_iter = 10))
])
pipeline_mlp.fit(train_vectors, y_train)
predicted_mlp_vectors = pipeline_mlp.predict(test_vectors)

mlp_vectors_accuracy = sklm.accuracy_score(y_test, predicted_mlp_vectors)

#**what is micro and stuff what are these precision and stuff
mlp_vectors_precision = sklm.precision_score(y_test, predicted_mlp_vectors,average='macro')
mlp_vectors_recall = sklm.recall_score(y_test, predicted_mlp_vectors,average='macro')
mlp_vectors_f1_score = sklm.f1_score(y_test, predicted_mlp_vectors,average='macro')
mlp_vectors_confusion_matrix = sklm.confusion_matrix(y_test,predicted_mlp_vectors)


#file = open("train_20news.txt","w") 
#file.write(str(train_20news)) 
#file.close() 
#
#file = open("test_20news.txt","w") 
#file.write(str(test_20news)) 
#file.close() 
#
#file = open("shuttled_20news.txt","w")  
#file.write(str(shuttled_20news)) 
#file.close() 