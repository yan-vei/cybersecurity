import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler 

# Here we have 2 sets of data that will remain constant during the whole training/testing process.
# Each dataset contains 5.000 samples (stratified) from the original dataset of Wednesday's traffic
# As we are targeting BENIGN traffic, BENIGN samples have label 1 and attacks have label 0

# To improve the performance of an SVM we have to use preprocessing, otherwise we will not
# be able to obtain reasonable results 

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

# Firstly we will run the SVC without any preprocessing and obtain mean results from several trials
# to see how the classifier behaves with the raw data
accuracies_no_prep = []
recalls_no_prep = []
precisions_no_prep = []
f1_scores_no_prep = []
running_time_no_prep = []

for i in range(5):
    clf = SVC(C=10, kernel='rbf', gamma='scale')
    start_time = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.time()

    accuracies_no_prep.append(accuracy_score(y_test, y_pred))
    recalls_no_prep.append(recall_score(y_test, y_pred))
    precisions_no_prep.append(precision_score(y_test, y_pred))
    f1_scores_no_prep.append(f1_score(y_test, y_pred))
    running_time_no_prep.append(end_time - start_time)

print("Results for the SVC without preprocessing:")
print("Mean accuracy: " + str(np.mean(accuracies_no_prep)))
print("Mean precision: " + str(np.mean(precisions_no_prep)))
print("Mean recalls: " + str(np.mean(recalls_no_prep)))
print("Mean F1-scores: " + str(np.mean(f1_scores_no_prep)))
print("Mean running time: " + str(np.mean(running_time_no_prep)) + "\n")


# Now for the SVC we will use preprocessing, MinMaxScaler in particular, to normalize the data
# and hopefully to obtain better results
# in order to compare our classifer with the KNN and the Decision Tree Classifier
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Here we append metrics of a SVC that we will use
accuracies_prep = []
recalls_prep = []
precisions_prep = []
f1_scores_prep = []
running_time_prep = []

# We are going to run the classifier 5 times and append average results to the arrays
for i in range(5):
    clf = SVC(C=10, kernel='rbf', gamma='scale')
    start_time = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.time()

    accuracies_prep.append(accuracy_score(y_test, y_pred))
    recalls_prep.append(recall_score(y_test, y_pred))
    precisions_prep.append(precision_score(y_test, y_pred))
    f1_scores_prep.append(f1_score(y_test, y_pred))
    running_time_prep.append(end_time - start_time)
   
print("Results for the SVC with preprocessing:")
print("Mean accuracy: " + str(np.mean(accuracies_prep)))
print("Mean precision: " + str(np.mean(precisions_prep)))
print("Mean recalls: " + str(np.mean(recalls_prep)))
print("Mean F1-scores: " + str(np.mean(f1_scores_prep)))
print("Mean running time: " + str(np.mean(running_time_prep)))

# As we can see, in general, preprocessing improves the results of a SVC classifier, especially
# accuracy and precision, even though that the computational times and metrics are still worse than those
# of KNN or Decision Tree

# Now we advance to the file SVM_gamma_and_C_selection.py to select the best pair of values
# of C and gamma