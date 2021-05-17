# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:08:48 2021

@author: Usuario
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv("data.csv")

dataset_sub = dataset.loc[1:600000].sample(1000)
 
X = np.nan_to_num(dataset_sub[dataset_sub.columns[0:78]])

#data = pd.read_csv('X', sep=',',header=None, usecols = [1,13,17])

index = 0
for i in X:
    plt.figure(index)
    plt.hist(X[index])

    plt.title(dataset_sub.columns[index])
    index += 1