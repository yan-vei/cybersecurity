# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:08:48 2021

@author: Usuario
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""1 13 17"""

dataset = pd.read_csv("dataCSV.csv")

dataset_sub = dataset.loc[1:600000].sample(1000)

"""pre_y = dataset_sub[' Label']
y = []
for i in pre_y:
    if (i == 'BENIGN'):
        y.append(0)
    else:
        y.append(1)
"""    
X = np.nan_to_num(dataset_sub[dataset_sub.columns[0:78]])

#data = pd.read_csv('X', sep=',',header=None, usecols = [1,13,17])

plt.figure(1)
plt.hist(X[1])
plt.title('Flow Duration')

plt.figure(2)
plt.hist(X[13])
plt.title('Bwd Packet Length Std')

plt.figure(3)
plt.hist(X[17])
plt.title('Flow IAT Std')




                
#data.plot(kind='bar')
#plt.ylabel('Frequency')
#plt.xlabel('Words')
#plt.title('Title')

#plt.show()