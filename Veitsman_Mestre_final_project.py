# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:45:07 2021

@authors: Yana & Marc
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

dataset = pd.read_csv("data.csv")

dataset_sub = dataset.loc[1:600000].sample(1000)

pre_y = dataset_sub[' Label']
y = []
for i in pre_y:
    if (i == 'BENIGN'):
        y.append(0)
    else:
        y.append(1)
        
# Check representativeness of sample in pandas (keeping proportions)
# Try one type of attacks vs others
    
    
X = np.nan_to_num(dataset_sub[dataset_sub.columns[0:78]])

scalerQuantile = QuantileTransformer(n_quantiles=100)
X_quantile = scalerQuantile.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_quantile, y, test_size = 0.3, random_state=0)

with_quantiles_KNeigbors = []
with_quantiles_rec = []
with_quantiles_pres = []

for i in range(1, 31):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train) 
    yPred = neigh.predict(X_test)
    with_quantiles_rec.append(recall_score(y_test,yPred))
    with_quantiles_pres.append(precision_score(y_test, yPred))

print("Recall")
print(with_quantiles_rec.index(max(with_quantiles_rec))+1)
print(max(with_quantiles_rec))
print("Precision:")
print(with_quantiles_rec.index(max(with_quantiles_rec))+1)
print(max(with_quantiles_pres))

scalerMinMax = MinMaxScaler()
X_min_max = scalerMinMax.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_min_max, y, test_size = 0.3, random_state = 0)

with_min_max_KNeighbors = []
with_min_max_rec = []
with_min_max_pres = []

for i in range(1, 31):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train) 
    yPred = neigh.predict(X_test)
    with_min_max_rec.append(recall_score(y_test,yPred))
    with_min_max_pres.append(precision_score(y_test, yPred))

print("Recall")
print(with_min_max_rec.index(max(with_min_max_rec))+1)
print(max(with_min_max_rec))
print("Precision:")
print(with_min_max_pres.index(max(with_min_max_pres))+1)
print(max(with_min_max_pres))

plt.plot(with_min_max_pres)
plt.ylim([.8,1])

plt.plot(with_min_max_rec)