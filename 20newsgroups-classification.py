import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

#fetch 20 newsgroups text dataset
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
class_names = twenty_train.target_names 
print (class_names) #prints all the categories

# Initialize the `tfidf_vectorizer` 
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(twenty_train.data) 
# Transform the test set 
tfidf_test = tfidf_vectorizer.transform(twenty_test.data)

clf = MultinomialNB()
clf.fit(tfidf_train, twenty_train.target)
pred = clf.predict(tfidf_test)

cm = confusion_matrix(twenty_test.target, pred)
print (cm)

score = accuracy_score(twenty_test.target, pred)
print("Accuracy:   %0.3f" % score)

f1 = f1_score(twenty_test.target, pred, average='macro')
print("F-1 Score:   %0.3f" % f1)




