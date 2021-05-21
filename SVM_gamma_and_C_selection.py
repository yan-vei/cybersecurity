import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler  
import matplotlib.pyplot as plt

# This file contains 3D-plots for the SVM classifier's metrics 
# In order to calculate the best possible pair of C and gamma values

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

# We can use preprocessing - MinMaxScaler - in particular to optimize the scales of the values of the features
# of the train and test sets
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#################################################

# We pepare several 3D-plots for the general metrics
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("Accuracies")
ax1.set_xlabel("C")
ax1.set_ylabel("Gamma")
ax1.set_zlabel("Accuracy")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Precisions")
ax2.set_xlabel("C")
ax2.set_ylabel("Gamma")
ax2.set_zlabel("Precision")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title("Recalls")
ax3.set_xlabel("C")
ax3.set_ylabel("Gamma")
ax3.set_zlabel("Recall")

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title("F1_scores")
ax4.set_xlabel("C")
ax4.set_ylabel("Gamma")
ax4.set_zlabel("F1_score")

# Trying medium values of C and gamma
for C_ in range(1,10,1):
    for gamma_ in range (1,10,1):
        # Prepare arrays for the metrics of the SVM classifier: accuracies, recalls, precisions
        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []
        for i in range(5):
            clf = SVC(C=C_, kernel='rbf', gamma=gamma_)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
            accuracies.append(accuracy_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
    
        print("C =" ,C_, "Gamma =",gamma_)               
        print("Mean accuracy: " + str(np.mean(accuracies)))
        print("Mean precision: " + str(np.mean(precisions)))
        print("Mean recalls: " + str(np.mean(recalls)))
        print("Mean F1-scores: " + str(np.mean(f1_scores)))
 
        ax1.scatter(C_,gamma_,np.mean(accuracies)) 
        ax2.scatter(C_,gamma_,np.mean(precisions)) 
        ax3.scatter(C_,gamma_,np.mean(recalls)) 
        ax4.scatter(C_,gamma_,np.mean(f1_scores)) 


plt.show()


#################################################

# Small ones

# Prepare plots
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("Accuracies")
ax1.set_xlabel("C")
ax1.set_ylabel("Gamma")
ax1.set_zlabel("Accuracy")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Precisions")
ax2.set_xlabel("C")
ax2.set_ylabel("Gamma")
ax2.set_zlabel("Precision")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title("Recalls")
ax3.set_xlabel("C")
ax3.set_ylabel("Gamma")
ax3.set_zlabel("Recall")

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title("F1_scores")
ax4.set_xlabel("C")
ax4.set_ylabel("Gamma")
ax4.set_zlabel("F1_score")


for C_ in range(1,3,1):
    for gamma_ in range (1,3,1):
        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []
        for i in range(5):
            gamma_=gamma_/100
            C_=C_/100
            clf = SVC(C=C_, kernel='rbf', gamma=gamma_)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
            accuracies.append(accuracy_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
    
        print("C =" ,C_, "Gamma =",gamma_)               
        print("Mean accuracy: " + str(np.mean(accuracies)))
        print("Mean precision: " + str(np.mean(precisions)))
        print("Mean recalls: " + str(np.mean(recalls)))
        print("Mean F1-scores: " + str(np.mean(f1_scores)))
 
        ax1.scatter(C_,gamma_,np.mean(accuracies)) 
        ax2.scatter(C_,gamma_,np.mean(precisions)) 
        ax3.scatter(C_,gamma_,np.mean(recalls)) 
        ax4.scatter(C_,gamma_,np.mean(f1_scores)) 


plt.show()

#################################################

# Big ones

# Prepare plots
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title("Accuracies")
ax1.set_xlabel("C")
ax1.set_ylabel("Gamma")
ax1.set_zlabel("Accuracy")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Precisions")
ax2.set_xlabel("C")
ax2.set_ylabel("Gamma")
ax2.set_zlabel("Precision")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title("Recalls")
ax3.set_xlabel("C")
ax3.set_ylabel("Gamma")
ax3.set_zlabel("Recall")

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title("F1_scores")
ax4.set_xlabel("C")
ax4.set_ylabel("Gamma")
ax4.set_zlabel("F1_score")


for C_ in range(1000,3000,1000):
    for gamma_ in range (1000,3000,1000):
        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []
        for i in range(5):
            clf = SVC(C=C_, kernel='rbf', gamma=gamma_)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
            accuracies.append(accuracy_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
    
        print("C =" ,C_, "Gamma =",gamma_)               
        print("Mean accuracy: " + str(np.mean(accuracies)))
        print("Mean precision: " + str(np.mean(precisions)))
        print("Mean recalls: " + str(np.mean(recalls)))
        print("Mean F1-scores: " + str(np.mean(f1_scores)))
 
        ax1.scatter(C_,gamma_,np.mean(accuracies)) 
        ax2.scatter(C_,gamma_,np.mean(precisions)) 
        ax3.scatter(C_,gamma_,np.mean(recalls)) 
        ax4.scatter(C_,gamma_,np.mean(f1_scores)) 


plt.show()
