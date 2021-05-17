# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:45:07 2021

@authors: Yana & Marc
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt  

dataset = pd.read_csv("data.csv")

# These are the original proportions in the dataset
proportions = {'DoS Hulk': 0.3335, 'DoS GoldenEye': 0.015, 'DoS SlowLoris': 0.0084, 
               'DoS SlowHttpTest': 0.00785, 'BENIGN': 0.6345, 'Heartbleed': 0.000016}

N = 50000 # number of samples to be sampled

# Obtaining a stratified sample according to the proportions
subset = dataset.groupby(' Label', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(dataset))))).sample(frac=1).reset_index(drop=True)

# Making sure that our sample is stratified
item_counts = subset[" Label"].value_counts(normalize=True)
print(item_counts)

# Appending labels according to the class that we are going to detect
y = []
for i in subset[' Label']:
    if i == 'DoS Slowhttptest':
        y.append(1)
    else:
        y.append(0)

# Extracting only features from the dataset and converting the values to a suitable numpy array representation              
X = subset[subset.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X = np.nan_to_num(X)

# Splitting our dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# We are going to check f1-scores to understand the weighted relationship between precision 
# and recall for each number of neighbors
f1_scores = []
precisions = []
recalls = []

# We will use KNeighbors classifier due to distance principle
# (objects of similar classes are located closely together)
for i in range (1, 15):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred, zero_division=1))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    

# Graphing f1-scores for the classifiers with different number of neighbors    
plt.plot([x for x in range (1,15)], f1_scores)
plt.xticks([x for x in range (1,15)])
plt.xlabel('Number of neighbors')
plt.ylabel('F1-score')
plt.title('F1-scores vs Number of Neighbors')
plt.show()

# Graphing precision and recall based on number of neighbors used
plt.plot([x for x in range (1,15)], precisions, label="Precisions")
plt.plot([x for x in range (1,15)], recalls, label="Recalls")
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Precision/Recall value')
plt.title('Precisions & Recalls vs Number of Neighbors')
plt.show()
   
    
