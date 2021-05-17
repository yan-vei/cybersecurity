# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:21:41 2021

@author: Yana
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt  

dataset = pd.read_csv("5kset.csv")

y = []
for i in dataset[' Label']:
    if i == 'HULK':
        y.append(1)
    else:
        y.append(0)

X = dataset[dataset.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X = np.nan_to_num(X)

accuracies = []
recalls = []
precisions = []
f1_scores = []

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = SVC()
    clf.fit(X, y)
    y_pred = clf.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
print("Mean accuracy: " + str(np.mean(accuracies)))
print("Mean precision: " + str(np.mean(precisions)))
print("Mean recalls: " + str(np.mean(recalls)))
print("Mean F1-scores: " + str(np.mean(f1_scores)))