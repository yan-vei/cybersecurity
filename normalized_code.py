# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:45:07 2021

@authors: Yana & Marc
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("data.csv")

# These are the original proportions in the dataset
proportions = {'DoS Hulk': 0.3335, 'DoS GoldenEye': 0.015, 'DoS SlowLoris': 0.0084, 
               'DoS SlowHttpTest': 0.00785, 'BENIGN': 0.6345, 'Heartbleed': 0.000016}

N = 10000 # number of samples to be sampled

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

# We will use RandomForestRegressor that doesn't require any preprocessing/scaling
# of data in order to weigh the features
regr = RandomForestRegressor(random_state=0)
regr.fit(X, y)
