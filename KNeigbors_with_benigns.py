# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:06:35 2021

@author: Yana
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt  

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Classifying BENIGNs as 1
y_train = train['Label']    
y_test = test['Label']

X_train = train[train.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_train = np.nan_to_num(X_train)

X_test = test[test.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_test = np.nan_to_num(X_test)

accuracies = []
recalls = []
precisions = []
f1_scores = []

for i in range (1, 11):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred, zero_division=1))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    accuracies.append(accuracy_score(y_test, y_pred))
    

# Graphing f1-scores for the classifiers with different number of neighbors    
plt.plot([x for x in range (1,11)], f1_scores)
plt.xticks([x for x in range (1,11)])
plt.xlabel('Number of neighbors')
plt.ylabel('F1-score')
plt.title('F1-scores vs Number of Neighbors')
plt.show()

# Graphing precision and recall based on number of neighbors used
plt.plot([x for x in range (1,11)], precisions, label="Precisions")
plt.plot([x for x in range (1,11)], recalls, label="Recalls")
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Precision/Recall value')
plt.title('Precisions & Recalls vs Number of Neighbors')
plt.show()