# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:21:41 2021

@author: Yana
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
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