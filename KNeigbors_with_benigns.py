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
from sklearn.preprocessing import MinMaxScaler 

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Classifying BENIGNs as 1
y_train = train['Label']    
y_test = test['Label']

X_train = train[train.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_train = np.nan_to_num(X_train)

X_test = test[test.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_test = np.nan_to_num(X_test)

accuracies_no_prep = []
recalls_no_prep = []
precisions_no_prep = []
f1_scores_no_prep = []

# First, let's see how K-NN neighbors behaves without any preprocessing
for i in range (1, 21):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    f1_scores_no_prep.append(f1_score(y_test, y_pred, zero_division=1))
    precisions_no_prep.append(precision_score(y_test, y_pred))
    recalls_no_prep.append(recall_score(y_test, y_pred))
    accuracies_no_prep.append(accuracy_score(y_test, y_pred))

# Now let's preprocess the training and testing data using MinMaxScaler from the sklearn library
# to see if the results improve
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

accuracies_prep = []
recalls_prep = []
precisions_prep = []
f1_scores_prep = []

# We repeat our tests and plot the same graphs for KNN-neigbors with preprocessing
for i in range (1, 21):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    f1_scores_prep.append(f1_score(y_test, y_pred, zero_division=1))
    precisions_prep.append(precision_score(y_test, y_pred))
    recalls_prep.append(recall_score(y_test, y_pred))
    accuracies_prep.append(accuracy_score(y_test, y_pred))

# Graphing f1-scores for the classifiers with different number of neighbors and preprocessing   
plt.plot([x for x in range (1,21)], f1_scores_no_prep, label = "F1 No Prep")
plt.plot([x for x in range (1,21)], f1_scores_prep, label = "F1 Prep")
plt.xticks([x for x in range (1,21)])
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1-score')
plt.title('F1-scores vs Number of Neighbors')
plt.show()

# Graphing precision and recall based on number of neighbors used and preprocessing
plt.plot([x for x in range (1,21)], precisions_prep, label="Precisions Prep")
plt.plot([x for x in range (1,21)], recalls_prep, label="Recalls Prep")
plt.plot([x for x in range (1,21)], precisions_no_prep, label="Precision No Prep")
plt.plot([x for x in range (1,21)], recalls_no_prep, label="Recall No Prep")
plt.xticks([x for x in range (1,21)])
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Precision/Recall value')
plt.title('Precisions & Recalls vs Number of Neighbors')
plt.show()

# Graphing accuracies based on number of neighbors and usage of preprocessing
plt.plot([x for x in range (1,21)], accuracies_prep, label="Accuracy Prep")
plt.plot([x for x in range (1,21)], accuracies_no_prep, label="Accuracy No Prep")
plt.xticks([x for x in range (1,21)])
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy value')
plt.title('Accuracy vs Number of Neighbors')
plt.show()