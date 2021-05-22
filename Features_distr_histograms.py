# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:08:48 2021

@author: Usuario
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer

dataset = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")

    
X = np.nan_to_num(dataset[dataset.columns[0:78]])


X_ = X.astype(np.float)

plt.figure(1)
plt.hist(X_[:,1])
plt.title('Flow Duration')

plt.figure(2)
plt.hist(X_[:,13])
plt.title('Bwd Packet Length Std')

plt.figure(3)
plt.hist(X_[:,17])
plt.title('Flow IAT Std')
