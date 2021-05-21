import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Here we have 2 sets of data that will remain constant during the whole training/testing process.
# Each dataset contains 5.000 samples (stratified) from the original dataset of Wednesday's traffic
# As we are targeting BENIGN traffic, BENIGN samples have label 1 and attacks have label 0
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Classifying BENIGNs as 1 and assigning labels to corresponding arrays
y_train = train['Label'].to_numpy()  
y_test = test['Label'].to_numpy()

# Assigning the rest of the data to the datasets, converting values of features
# to float32 and to numpy arrays (sklearn by default uses 32 bits precision)
X_train = train[train.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_train = np.nan_to_num(X_train)

X_test = test[test.columns[0:-1]].astype(dtype=np.float32).to_numpy()
X_test = np.nan_to_num(X_test)

# Prepare arrays to store metrics, such as accuracies, recalls,
# precisions, f1_scores and running times for each tree
accuracies = []
recalls = []
precisions = []
f1_scores = []
running_time = []

# Prepare a dataframe to save and later compare results of running tree several times
df = {'Running time': running_time, 'Accuracy': accuracies, 'Recall': recalls, 
      'Precision': precisions, 'F1-score': f1_scores}

# To get the average metrics, such as accuracy, recalls, etc.
# We will run the classifier several times and then we will calculate
# averages based on what we obtain
for i in range(20):
    tree = DecisionTreeClassifier()
    start_time = time.time()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    end_time = time.time()
    
    accuracies.append(accuracy_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    running_time.append(end_time - start_time)
    
print("Mean accuracy: " + str(np.mean(accuracies)))
print("Mean precision: " + str(np.mean(precisions)))
print("Mean recalls: " + str(np.mean(recalls)))
print("Mean F1-scores: " + str(np.mean(f1_scores)))
print("Mean running time:" + str(np.mean(running_time)))

# Save our statistics to a dataframe (if we later need to export it to a more 
# readable format, like an excel or csv file)
pd.DataFrame(df)

# As we see, the tree is, in fact, the best classifier for this problem. It has the fastest
# running time, the best metrics' values, like accuracy, precision and recalls,
# and doesn't require any preprocessing of data


