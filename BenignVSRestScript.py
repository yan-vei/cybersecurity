import pandas as pd
import numpy as np

# This script was created in order to obtain stratified samples of 5.000 samples
# for training and testing from the whole Wednesday dataset of about 690.000 samples

# In this dataset we are going to label BENIGNs as 1s and all the attacks as 0s

dataset = pd.read_csv("data.csv")

# Obtaining training data

N = 5000 # number of samples

# stratified set of N samples
subset = dataset.groupby(' Label', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(dataset))))).sample(frac=1).reset_index(drop=True)

# labels
y = []
for i in subset[' Label']:
    if i == 'BENIGN':
        y.append(1)
    else:
        y.append(0)

# We create a new frame to later save to a .csv file
data = subset[subset.columns[0:-1]]
data['Label'] = y

data.to_csv('train.csv')

# Obtaining testing data
N = 5000 # number of samples

# stratified set of N samples
subset = dataset.groupby(' Label', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(dataset))))).sample(frac=1).reset_index(drop=True)

# labels
y = []
for i in subset[' Label']:
    if i == 'BENIGN':
        y.append(1)
    else:
        y.append(0)

# We create a new frame to later save to a .csv file
data = subset[subset.columns[0:-1]]
data['Label'] = y

data.to_csv('test.csv')