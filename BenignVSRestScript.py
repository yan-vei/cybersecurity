# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:57:31 2021

@author: Yana
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("data.csv")

# obtaining training data

N = 5000

subset = dataset.groupby(' Label', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(dataset))))).sample(frac=1).reset_index(drop=True)

y = []
for i in subset[' Label']:
    if i == 'BENIGN':
        y.append(1)
    else:
        y.append(0)

data = subset[subset.columns[0:-1]]
data['Label'] = y

data.to_csv('train.csv')

# obtaining testing data
N = 5000

subset = dataset.groupby(' Label', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(dataset))))).sample(frac=1).reset_index(drop=True)

y = []
for i in subset[' Label']:
    if i == 'BENIGN':
        y.append(1)
    else:
        y.append(0)

data = subset[subset.columns[0:-1]]
data['Label'] = y

data.to_csv('test.csv')