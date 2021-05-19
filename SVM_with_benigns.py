# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:04:22 2021

@author: Yana
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler  

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Classifying BENIGNs as 1
y_train = train['Label'].to_numpy()  
y_test = test['Label'].to_numpy()

X_train = train[train.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_train = np.nan_to_num(X_train)

X_test = test[test.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_test = np.nan_to_num(X_test)

# We can use preprocessing - MinMaxScaler - in particular to optimize the scales of the values of the features
# of the train and test sets
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

accuracies = []
recalls = []
precisions = []
f1_scores = []

for i in range(5):
    clf = SVC(C=10, kernel='rbf', gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    
print("Mean accuracy: " + str(np.mean(accuracies)))
print("Mean precision: " + str(np.mean(precisions)))
print("Mean recalls: " + str(np.mean(recalls)))
print("Mean F1-scores: " + str(np.mean(f1_scores)))